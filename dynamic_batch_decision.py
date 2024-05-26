#!/usr/bin/env Python
# coding=utf-8

import numpy as np
# import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import argparse
import logging
import sys
from tensorboardX import SummaryWriter
# import  matplotlib.pyplot as plt

# from init_coil_data import Get_Batch_Data
from dataset import CHDataset
from pointer_network import PointerNet, PointerNetLoss


if __name__ == "__main__":
  # Parse argument
    parser = argparse.ArgumentParser("dynamic_batch_decision")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--bz", type=int, default=32)
    # 设置1000是因为有些板卷在当次不能被决策，会留到下一次决策，那么seq_len可能远远大于10
    parser.add_argument("--max_in_seq_len", type=int, default=1000)
    parser.add_argument("--max_out_seq_len", type=int, default=21)
    parser.add_argument("--rnn_hidden_size", type=int, default=128)
    parser.add_argument("--attention_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--beam_width", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_norm", type=float, default=5.)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument("--check_interval", type=int, default=20)
    parser.add_argument("--nepoch", type=int, default=10000)
    parser.add_argument("--train_filename", type=str, default=("./train_data/slab-train-data.txt", "./train_data/order-train-data.txt"))
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    # Pytroch configuration
    if args.gpu >= 0 and torch.cuda.is_available():
        args.use_cuda = True
        torch.cuda.device(args.gpu)
    else:
        args.use_cuda = False
    # Logger
    logger = logging.getLogger("Convex Hull")
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    # Summary writer
    writer = SummaryWriter(args.log_dir)

    #Init batch data
    # Get_Batch_Data(10000)

    # Loading data
    # 在pytorch中使用自己的数据集，需要创建Dataset类，类的要求见：https://blog.csdn.net/kahuifu/article/details/108654421
    train_ds = CHDataset(args.train_filename, args.max_in_seq_len,
                       args.max_out_seq_len)
    logger.info("Train data size: {}".format(len(train_ds)))

    # num_workers启动多少子进程，默认为0启动主进程
    # drop_last(bool)：True如果最后一个batch_size不能被数据集整除，则删除后一个批次。False 最后一个batch_size变小
    train_dl = DataLoader(train_ds, num_workers=2, batch_size=args.bz, drop_last=True)

    # Init model
    model = PointerNet("LSTM",  #rnn_type = "lstm"
        False,      # bidirectional = False
        args.num_layers,  # num_layers = args.num_layers =1
        (2, 3),  # input_size 板坯数据2维 合同数据3维
        args.rnn_hidden_size,  # rnn_hidden_size = args.rnn_hidden_size =128
        0.0
                       )  # dropout = 0.0 (概率)
    criterion = PointerNetLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # model = torch.load("net_para_0.pth")
    #将模型上的所有参数挂在GPU上计算
    # if args.use_cuda:
    #   print(args.use_cuda)
    #   model.cuda()

    #查看模型所有的参数
    #for param in model.named_parameters():
    #  print(param[0], param[1].requires_grad)

    unuse_coil_info = {}
    unuse_ord_info = {}
    use_ord_info = {}
    reward_plot = list()

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.nepoch):
            # model.train() ：启用 BatchNormalization 和 Dropout
            # model.eval() ：不启用 BatchNormalization 和 Dropout
            model.train()
            total_loss = 0.
            batch_cnt = 0.
            r_epoch = list()
            p_epoch = list()

            for date, (s_inp, s_inp_len, o_inp, o_inp_len) in enumerate(train_dl):

                # date: index, 表示板卷倒料时间，用于reward计算板卷脱期成本
                # b_inp.shape(torch.float32)= [batch_size, max_in_seq_len, input_size] = [3, 1000, 7]
                # b_inp_len(torch.int32) = [batch_size, 1] = [3,1]

                # 将数据挂在GPU上，注：如果model.cuda()已经挂在了GPU上，相应的，输入数据也需要挂在GPU上保持一致
                #   if args.use_cuda:
                #       date = torch.tensor(date).cuda()
                #       s_inp = s_inp.cuda()
                #       s_inp_len = s_inp_len.cuda()
                #       o_inp = o_inp.cuda()
                #       o_inp_len = o_inp_len.cuda()
                r = [0.0] * args.bz     # 初始化奖励为0
                probs = list()
                for iter in range(1):   # 迭代次数暂时设置为5
                    # 处理板坯数据
                    if date % 30 != 0 or (date % 30 == 0 and iter != 0):

                        for batch, key in enumerate(unuse_coil_info):

                            if iter != 0:
                                # 由dic获得的数据没有[0 0]标志位，这里将长度变量从0变1，相当于在开头加了标志位，
                                # 因为inp为长度固定的全零量
                                s_inp_len[batch, :] = 1

                            for index in range(len(unuse_coil_info[key])):  # 只加入未被使用的板坯
                                # .long()的作用是因为tensor作为索引时必须为Long类型
                                # 将unuse_coil_info中板卷加入到此次的板卷输入中
                                # b_inp的长度为1000，有效部分的长度存储在b_inp_len中，加入unuse_coil_info中的板卷时，
                                # 只需要直接修改被填充为全0的tensor，而不是将此板卷添加在末尾。
                                s_inp[batch, s_inp_len[batch, :].long()+index, :] = torch.clone(unuse_coil_info[key][index])

                            # 改变input的长度，否则会让encoding做padding有问题
                            s_inp_len[batch, :] = s_inp_len[batch, :].clone() + len(unuse_coil_info[key])

                    # 处理合同数据
                    if date % 30 != 0 and iter == 0:

                        for batch, key in enumerate(unuse_ord_info):

                            count_idx = o_inp_len[batch, :].long().clone()

                            for index in range(len(unuse_ord_info[key])):

                                if unuse_ord_info[key][index][2] == 0:  # 排除到期的合同
                                    continue

                                o_inp[batch, count_idx, :] = unuse_ord_info[key][index]
                                # 改变input的长度，否则会让encoding做padding有问题
                                # o_inp_len[batch, :] += 1
                                count_idx += 1
                            o_inp_len[batch, :] = count_idx

                        for batch, key in enumerate(use_ord_info):

                            count_idx = o_inp_len[batch, :].long().clone()

                            for index in range(len(use_ord_info[key])):

                                if use_ord_info[key][index][2] == 0:    # 排除到期的合同
                                    continue
                                o_inp[batch, count_idx, :] = use_ord_info[key][index]
                                # 改变input的长度，否则会让encoding做padding有问题
                                # o_inp_len[batch, :] += 1
                                count_idx += 1
                            o_inp_len[batch, :] = count_idx

                    elif iter > 0:    # 合同的个数不变，只有欠量被更新过
                        for batch, key in enumerate(unuse_ord_info):
                            o_inp_len[batch, :] = 1
                            for index in range(len(unuse_ord_info[key])):
                                o_inp[batch, o_inp_len[batch, :].long()+index, :] = torch.clone(unuse_ord_info[key][index])

                            # 改变input的长度，否则会让encoding做padding有问题
                            o_inp_len[batch, :] += len(unuse_ord_info[key])
                        for batch, key in enumerate(use_ord_info):
                            for index in range(len(use_ord_info[key])):
                                o_inp[batch, o_inp_len[batch, :].long()+index, :] = torch.clone(use_ord_info[key][index])

                            # 改变input的长度，否则会让encoding做padding有问题
                            o_inp_len[batch, :] = o_inp_len[batch, :].clone() + len(use_ord_info[key])

                    R, action_probs1, action_probs2, actions1, actions2, action_idxs1, action_idxs2,\
                        unuse_coil_dic1, unuse_coil_dic2, use_ord_dic = model(date, s_inp, s_inp_len, o_inp, o_inp_len)
                    r = [r[k] + float(R[k]) for k in range(args.bz)]
                    probs += (action_probs1 + action_probs2)

                    unuse_coil_info = unuse_coil_dic1
                    unuse_ord_info = unuse_coil_dic2
                    use_ord_info = use_ord_dic


                    for batch in range(args.bz):
                        # 合同数据长为3， 板坯数据长为2
                        for index in range(s_inp_len[batch][0]):
                            s_inp[batch, index, :] = torch.zeros(2, dtype=torch.float32)

                        for index in range(o_inp_len[batch][0]):
                            o_inp[batch, index, :] = torch.zeros(3, dtype=torch.float32)
                    s_inp_len = torch.tensor([[0] for num_b in range(args.bz)], dtype=torch.int32)
                    o_inp_len = torch.tensor([[0] for num_b in range(args.bz)], dtype=torch.int32)

                # 计算剩余板坯库存费用
                for batch, key in enumerate(unuse_coil_info):
                    batch_weight = sum([slab[0] for slab in unuse_coil_info[key]])
                    #r[batch] -= float(batch_weight * 2.0)      # 库存费用为2.0
                    r[batch] -= float(batch_weight * 1.0)      # 库存费用为1.0
                # 更新合同交货期、计算交货合同欠量惩罚
                for batch, key in enumerate(unuse_ord_info):
                    for index in range(len(unuse_ord_info[key])):
                        unuse_ord_info[key][index][2] -= 1
                    for index in range(len(use_ord_info[key])):
                        use_ord_info[key][index][2] -= 1
                    weight_remain = sum([order[0] for order in use_ord_info[key] if order[2] == 0])
                    weight_remain += sum([order[0] for order in unuse_ord_info[key] if order[2] == 0])
                    r[batch] -= float(weight_remain * 2.0)     # 欠量惩罚为2.0

                # print("r: ", r)
                r_epoch.append(r)   # 本次epoch的奖励值（两个batch）

                logprobs = 0
                for prob in probs:
                    logprob = torch.log(prob)
                    logprobs = logprobs + logprob
                # print(logprobs)
                logprobs[logprobs < -1000] = 0.
                #print(logprobs)
                # print("logprobs: ", logprobs)
                p_epoch.append(logprobs)    # 本次epoch的logP值（两个batch)

                # break
                  # 计算目标值，训练网络
                if (date + 1) % 30 != 0:
                    continue
                R_reinforce = torch.tensor([0.0] * args.bz, dtype=torch.float32)
                policy_loss = []
                rewards = []
                #r_epoch = torch.tensor(r_epoch, dtype=torch.float32, requires_grad=False, device = 'cuda:0')
                for r in r_epoch[::-1]:
                    r = torch.tensor(r, dtype=torch.float32, requires_grad=False
                                    # , device='cuda:0'
                                     )
                    # print(r)
                    # print(R_reinforce)
                    R_reinforce = (R_reinforce * args.gamma + r / 100)
                    rewards.insert(0, R_reinforce)
                for log_prob, reward in zip(p_epoch, rewards):
                    policy_loss.append(-log_prob * reward)

                optimizer.zero_grad()
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                reward_plot.append(sum(map(sum, r_epoch)))

                if (date+1) % 30 == 0:
                    print("epoch = %d,  average reward = %.3f"  %   (epoch,  sum(map(sum, r_epoch)) / (args.bz * 30)))
                del r_epoch[:]
                del p_epoch[:]
                # plt.plot(reward_plot)
                # plt.pause(0.001)

                # reinforce = advantage * logprobs
                # reinforce = torch.tensor(advantage, requires_grad=False) * logprobs
                #print("reinforce = ", reinforce)
                #print("logprobs = ", logprobs)
                # loss = reinforce.mean()
                # print("loss = ", loss)

                # optimizer.zero_grad()
                # loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                  float(args.max_grad_norm), norm_type=2)
                optimizer.step()


                # plt.plot(reward_plot)
                # plt.show()

            if epoch % 50 == 0:
                print("save parameters")
                torch.save(model, "LOGs/net_para_" + str(epoch) + ".pth")








