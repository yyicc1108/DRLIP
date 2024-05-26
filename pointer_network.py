#!/usr/bin/env Python
# coding=utf-8

import torch
from torch import nn
from torch.autograd import Variable

from layers.seq2seq.encoder import RNNEncoder
from layers.seq2seq.decoder import RNNDecoderBase, PointerNetRNNDecoder
from layers.attention import Attention, sequence_mask
from MatchMultiKnap import MatchSolver


class PointerNet(nn.Module):
    """ Pointer network
    Args:
      rnn_type (str) : rnn cell type                ="LSTM"
      bidirectional : whether rnn is bidirectional  =True
      num_layers : number of layers of stacked rnn  =1
      encoder_input_size : input size of encoder    =4
      rnn_hidden_size : rnn hidden dimension size   =128
      dropout : dropout rate                        =0.0
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 encoder_input_size, rnn_hidden_size, dropout):
        super(PointerNet, self).__init__()
        # encoder1 decoder1 对应slab, encoder2 decoder2 对应order
        self.encoder1 = RNNEncoder(rnn_type, bidirectional,
                                  num_layers, encoder_input_size[0], rnn_hidden_size, dropout)
        self.decoder1 = PointerNetRNNDecoder(rnn_type, bidirectional,
                                            num_layers, encoder_input_size[0], rnn_hidden_size, dropout, is_slab_=True)
        self.encoder2 = RNNEncoder(rnn_type, bidirectional,
                                  num_layers, encoder_input_size[1], rnn_hidden_size, dropout)
        self.decoder2 = PointerNetRNNDecoder(rnn_type, bidirectional,
                                            num_layers, encoder_input_size[1], rnn_hidden_size, dropout, is_slab_=False)
        self.fc1 = nn.Linear(rnn_hidden_size * 2, rnn_hidden_size)
        self.fc2 = nn.Linear(rnn_hidden_size * 2, rnn_hidden_size)

    def forward(self, date, inp1, inp_len1, inp2, inp_len2):
        '''
        Args:
          date：int,板卷倒料日期
          inp(torch.float32): [batch_size, max_in_seq_len, input_size] = [batch_size, 1001, 7]
          inp_len(torch.int32): [batch_size, 1]

        '''
        batch_num = inp1.shape[0]
        # transpose(0, 1) 交换第0个维度和第一个维度
        # 注意，若RNN中设置了batch_first=True的话  则需要batch_size在前面，此时无需交换两个维度
        inp1 = inp1.transpose(0, 1)  # [batch_size, max_in_seq_len, input_size] -> [max_in_seq_len, batch_size, input_size]
        inp2 = inp2.transpose(0, 1)

        # encoder可以处理句子变长的问题，可以将max_in_seq_len转换成了转换成实际的最大长度
        # max_in_seq_len转换成了----->>>>此seq_len为同一batch中的句子实际最大长度
        # encoder_output.shape = [seq_len, batch_size, hidden_size]
        # hidden = (h_t, c_t), h_t.shape=[layer_num, batch_size, hidden_size]  c_t=[1, 2, 128]
        encoder_output1, hidden1 = self.encoder1(inp1, inp_len1)
        encoder_output2, hidden2 = self.encoder2(inp2, inp_len2)

        ht_1_2 = torch.cat((hidden1[0], hidden2[0]), dim=2)
        ct_1_2 = torch.cat((hidden1[1], hidden2[1]), dim=2)
        h_t = nn.functional.relu(self.fc1(ht_1_2))
        c_t = nn.functional.relu(self.fc2(ct_1_2))

        action_probs1, actions1, action_idxs1, unuse_coil_dic1, use_slab_dic = \
            self.decoder1(date, inp1, encoder_output1, (h_t, c_t), inp_len1)  # inp用于decoder_input

        action_probs2, actions2, action_idxs2, unuse_coil_dic2, use_order_dic = \
            self.decoder2(date, inp2, encoder_output2, (h_t, c_t), inp_len2)  # inp用于decoder_input


        R = list()
        for i in range(batch_num):
            # 板坯集合非空，合同集合为空，则相当于全部配给自拟合同
            # 合同集合非空，板坯集合为空，则不匹配
            if len(use_slab_dic["batch" + str(i)]) == 0:
                R.append(0.0)
                continue
            if len(use_order_dic["batch" + str(i)]) == 0:
                R.append(sum([slab_[0] for slab_ in use_slab_dic["batch" + str(i)]]) * (0.1))  # 全部匹配给自拟合同
                continue
            solver = MatchSolver(use_slab_dic["batch" + str(i)], use_order_dic["batch" + str(i)])
            match_result, obj_value = solver.solve_model()
            R.append(obj_value)
            # 更新合同欠量，只需更新被选中的合同的欠量即可
            for ord_idx in range(len(use_order_dic['batch' + str(i)])):
                use_order_dic['batch' + str(i)][ord_idx][0] -= \
                    sum([match_result[ord_idx][s_idx] * use_slab_dic["batch" + str(i)][s_idx][0]
                         for s_idx in range(len(match_result[ord_idx]))])
        '''print("use_slab_dic", use_slab_dic)
        print("unuse_coil_dict", unuse_coil_dic1)'''

        return R, action_probs1, action_probs2, actions1, actions2,\
            action_idxs1, action_idxs2, unuse_coil_dic1, unuse_coil_dic2, use_order_dic


class PointerNetLoss(nn.Module):
    """ Loss function for pointer network
    """

    def __init__(self):
        super(PointerNetLoss, self).__init__()

    def forward(self, target, logits, lengths):
        """
        Args:
          target : label data (batch_size, tgt_max_len)             (batch_size, 12)
          logits : predicts (batch_size, tgt_max_len, src_max_len)  (batch_size, 12, 11)
          lengths : length of label data (batch_size)  (3,1)
        """
        '''print(target.shape)
        print(logits.shape)
        print(lengths.shape)'''

        '''print(lengths)  # 6 9 7'''
        _, tgt_max_len = target.size()  # tgt_max_len=7
        logits_flat = logits.view(-1, logits.size(-1))  # logits_flat.shape = [batch_size*12, 11]
        log_logits_flat = torch.log(logits_flat)
        target_flat = target.view(-1, 1)  # target_flat.shape = [batch_size*12, 1]

        '''print(log_logits_flat)
        print(target_flat)'''

        losses_flat = -torch.gather(log_logits_flat, dim=1, index=target_flat.long())  # 按照索引将对应索引位置的值取出来

        '''print(losses_flat)
        print(*target.size())
        print(target.size())'''
        losses = losses_flat.view(*target.size())
        '''print(losses)'''
        mask = sequence_mask(lengths, tgt_max_len)
        '''print(mask)'''
        mask = Variable(mask)
        '''print(mask.float())'''
        losses = losses * mask.float()
        '''print(losses)
        print(losses.sum())
        print(lengths.float().sum())'''
        loss = losses.sum() / lengths.float().sum()
        return loss