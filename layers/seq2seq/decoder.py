#!/usr/bin/env Python
# coding=utf-8

import math
import numpy as np
import operator
import torch
from torch import nn
import torch.nn.functional as F

from layers.seq2seq.encoder import rnn_factory
from layers.attention import Attention
from layers.attention1 import Attention1

class RNNDecoderBase(nn.Module):
  """ RNN decoder base class
  Args:
    rnn_type : rnn cell type, ["LSTM", "GRU", "RNN"]    ="LSTM"
    bidirectional : whether use bidirectional rnn       =True
    num_layers : number of layers in stacked rnn        =1
    input_size : input dimension size                   =2
    hidden_size : rnn hidden dimension size             =128
    dropout : dropout rate                              =0.0
  """
  def __init__(self, rnn_type, bidirectional, num_layers, input_size, hidden_size, dropout):
    super(RNNDecoderBase, self).__init__()
    if bidirectional:
      assert hidden_size % 2 == 0
      hidden_size = hidden_size // 2
    self.rnn, _ = rnn_factory(rnn_type,
      input_size=input_size,
      hidden_size=hidden_size,
      bidirectional=bidirectional,
      num_layers=num_layers,
      dropout=dropout)
  
  def forward(self, tgt, encoder_output, hidden, memory_lengths=None):
    """
    Args:
      tgt: target sequence
      encoder_output : memory from encoder or other source
      hidden : init hidden state
      memory_lengths : lengths of memory
    """
    raise NotImplementedError()


class PointerNetRNNDecoder(RNNDecoderBase):
  """
  Pointer network RNN Decoder, process all the output together
  Args:
    rnn_type : rnn cell type, ["LSTM", "GRU", "RNN"]    ="LSTM"
    bidirectional : whether use bidirectional rnn       =True
    num_layers : number of layers in stacked rnn        =1
    input_size : input dimension size                   =4
    hidden_size : rnn hidden dimension size             =128
    dropout : dropout rate                              =0.0
  """

  def __init__(self, rnn_type, bidirectional, num_layers,
               input_size, hidden_size, dropout, is_slab_):
    super(PointerNetRNNDecoder, self).__init__(rnn_type, bidirectional, num_layers,
                                               input_size, hidden_size, dropout)
    self.attention = Attention1("Bahdanau", hidden_size)

    self.input_size = input_size

    self.is_slab = is_slab_

    self.curve_61 = torch.tensor([1,0], dtype=torch.int64)
    self.curve_99 = torch.tensor([0,1], dtype=torch.int64)

  def apply_mask_to_logits(self, logits, mask, idxs):
    '''
    idxs：将上一次被随机采样到的数字设置到-inf
    :param logits(torch.float32):  [batch_size, seq_len]
    :param mask(torch.unit8):    [batch_size, seq_len]
    :param idxs (torch.int64): [batch_size]
    :return:
    '''

    batch_size = logits.size(0)
    seq_len = logits.size(1)
    clone_mask = mask.clone()

    # 将上一次被随机采样到的数字idxs设置到-inf
    if idxs is not None:
      clone_mask[[i for i in range(batch_size)], idxs.data] = 1

      logits[clone_mask.bool()] = -np.inf

    #  以下三行代码的逻辑是：
    #       由于同一个batch中，每个句子不等长，所以在（不放回）抽样时，较短的句子可能很早便都抽没了
    #       因此，短句子的字符抽没时，以后抽样均指定第一个start_token，即概率设置为1，其余的都为-inf

    # 某行的数据是否全部都为-inf，即是否-inf的个数等于seq_len
    boolvec = logits.eq(-float('inf')).int().sum(axis=1) == seq_len
    # 判断是否有某行的数据是否全部都为-inf的情况存在，即存在True，则运行赋值
    # 此处if起到的作用是加快速度，若不满足条件，这赋值语句不用执行
    if boolvec.int().sum() >= 1:
      logits[boolvec, 0] = 1

    # 2020年末实现的方式，摒弃掉for循环(效率低)，采用上述三行代码实现的形式
    # for index in range(batch_size):
    #   if torch.eq(logits[index, :], -float('inf')).int().sum() == seq_len:
    #     logits[index, 0] = 1

    return logits, clone_mask

  def cal_reward_(self, coil_list, curve_61_flag=True, mixMatchFlag=False):

    if mixMatchFlag == False:
      if len(coil_list) == 0:
        return -10

      # 确定炉子数
      complete = (len(coil_list) // 3)

      weight_reward = 0  #重量收益
      width_reward = 0  #高度收益
      date_reward = 0   #满足交期收益
      date_delay_penalty = 0 # 拖期惩罚

      if curve_61_flag == True:
        for index in range(len(coil_list)):
          weight_reward =  weight_reward + coil_list[index][0]  # 炉子总重量收益
          width_reward = width_reward + coil_list[index][1]   # 板卷总高度收益
          if coil_list[index][4] >=1:
            date_reward = date_reward + 60          #正数
          if coil_list[index][4] < 1:
            date_delay_penalty = date_delay_penalty + (coil_list[index][4] - 1) * (-80)  # 正数

        # # *2 /25 的目标是控制重量和宽度惩罚处在同一数量级
        weight_penalty = (80*complete - weight_reward) * 2
        width_penalty = (3750*complete - width_reward) / 25

        reward = date_reward - (date_delay_penalty + weight_penalty + width_penalty)

        return reward

      else:
        for index in range(len(coil_list)):
          weight_reward = weight_reward + coil_list[index][0]  # 炉子总重量收益
          width_reward = width_reward + coil_list[index][1]  # 板卷总高度收益

          if coil_list[index][4] >= 1:
            date_reward = date_reward + 60  # 正数
          if coil_list[index][4] < 1:
            date_delay_penalty = date_delay_penalty + (coil_list[index][4] - 1) * (-120)  # 正数

        weight_penalty = (80 * complete - weight_reward) * 1  # *2 /25 的目标是控制重量和宽度惩罚处在同一数量级
        width_penalty = (3750 * complete - width_reward) / 40

        reward = date_reward - (date_delay_penalty + weight_penalty + width_penalty)

        return reward


    if mixMatchFlag == True:

      if len(coil_list) == 0:
        return 0

      # 商和余数  三个板卷为一个炉子
      complete = (len(coil_list) // 3)
      remainder = len(coil_list) % 3
      if remainder != 0:  # 等于0说明炉子装的正好，不等于0需要额外一个炉子
        complete += 1

      weight_reward = 0  # 重量收益
      width_reward = 0  # 高度收益
      date_reward = 0  # 满足交期收益
      date_delay_penalty = 0  # 拖期惩罚

      for index in range(len(coil_list)):
        weight_reward = weight_reward + coil_list[index][0]  # 炉子总重量收益
        width_reward = width_reward + coil_list[index][1]  # 板卷总高度收益

        if coil_list[index][4] >= 1:
          date_reward = date_reward + 60  # 正数
        if coil_list[index][4] < 1:
          date_delay_penalty = date_delay_penalty + (coil_list[index][4] - 1) * (-120)  # 正数

      weight_penalty = (80 * complete - weight_reward) * 1  # *2 /25 的目标是控制重量和宽度惩罚处在同一数量级
      width_penalty = (3750 * complete - width_reward) / 40
      mixmatch_penalty = 100

      reward = date_reward - (date_delay_penalty + weight_penalty + width_penalty + mixmatch_penalty)

      return reward


  def reward(self, date, input, action_idxs, actions):
    '''
    计算奖励
    :param date:
    :param input(torch.float32): shape = [max_in_seq_len, batch_size, input_size]
    :param action_idxs (list):  被选择动作的index（按顺序）
    :param actions (list): 每个动作的具体信息（动作本身）
    :return:
    '''

    batch_size = input.shape[1]

    # 将list中的tensor拼接，然后变化维度[batch_size,seq_len]
    # action_idxs_.shape = [batch_size, seq_len]
    action_idxs_ = torch.cat(action_idxs, dim=0).reshape(-1, batch_size).transpose(1, 0)
    batch_size = action_idxs_.size(0)
    seq_len = action_idxs_.size(1)

    # print("action_idxs_ = ", action_idxs_)


    # use_coil_dic用于计算reward, unuse_coil_dic用于存储未被使用的板卷信息，并传给下一次输入信息
    # 列表推导式生成字典，字典的键值对为 batch0 - [](空列表)
    # 思考？： 这地方为什么要用字典？ 考虑到使用和未使用的板卷的每个batch个数可能是不同的，对tensor来说会造成维度不同的情况， 或许有更好的方式？
    use_coil_dic = {'batch' + str(i): [] for i in range(batch_size)}
    unuse_coil_dic = {'batch' + str(i): [] for i in range(batch_size)}

    # 利用action_idxs_中的第一个0位置，把板卷分为use_coil_dic和use_coil_dic
    for batch in range(batch_size):  # 遍历每行 即 batch0  batch1...
      for seq in range(seq_len):     # 遍历每个动作
       if action_idxs_[batch, seq] == 0:    # 判断是否遇到了第一个0 （0代表终止符） 0前面的代表板卷在本次决策被选择进行生产
        un_seq = seq+1    # 索引值+1 主要是为了索引未被使用的板卷

        for c in range(seq):  # range(seq) = 0,1,...,seq-1
          use_coil_dic['batch' + str(batch)].append(input[action_idxs_[batch, c], batch, :].clone())

        for uc in range(seq_len - seq - 1):  #未被使用板卷的个数 ： seq_len - seq - 1 = 总数-已使用-标识位0
          if action_idxs_[batch, un_seq] != 0:
            unuse_coil_dic['batch' + str(batch)].append(input[action_idxs_[batch, un_seq], batch, :].clone())
            un_seq = un_seq + 1

        break  #目的就是找第一个0所在的位置，找到之后后续不必再进行for循环
    '''
    if not self.is_slab:    # 合同生产时长-1
      # 开始处理未被使用的板卷：
      # 所有未被使用的板卷交货期都需要-1，然后返回给input，作为新的输入即可，每次use_coil_dic和unuse_coil_dic都是新的)
      for key in unuse_coil_dic:
        if len(unuse_coil_dic[key]) == 0:  # 如果list中元素个数为0，那么直接跳过不必处理 否则下面torch.stark方法会报错
          unuse_coil_dic[key] = torch.tensor(unuse_coil_dic[key])
          continue
        # 将装有tensor的list 转为tensor
        # 去除重复的行， reshape的作用是确保[:,self.input_size]切片有效
        # https://blog.csdn.net/liu16659/article/details/114752918
        unuse_coil_dic[key] = torch.stack(unuse_coil_dic[key], 0).reshape(-1, self.input_size)
        # 因本次未能生产，将unuse_coil_dic中的板卷交期都-1
        unuse_coil_dic[key][:, self.input_size-1] = unuse_coil_dic[key][:, self.input_size-1] - 1
    '''
    # print(use_coil_dic)
    # print(unuse_coil_dic)
    '''
    # 开始处理已使用板卷：
    # 先将板卷按照退火曲线号分类，然后分别计算回报奖励R
    R = []
    for batch, key in enumerate(use_coil_dic):

      if len(use_coil_dic[key]) == 0:  # 如果list中元素个数为0，那么直接跳过不必处理 否则下面torch.stark方法会报错
        continue

      coils_61 = []
      coils_99 = []

      # 将list通过stack操作变成tensor（[num_use_coils, input_size]） num_use_coils为已使用板卷的数量
      use_coil_dic[key] = torch.stack(use_coil_dic[key], 0).reshape(-1, self.input_size)

      # 按照退火曲线号进行分类
      for index in range(use_coil_dic[key].shape[0]):

        coil_curve = use_coil_dic[key][index,2:4].long()  # tensor.long() = tensor.int64

        if coil_curve.equal(self.curve_61):     # tensor.equal是判断两个tensor是否相等，只返回一个bool值， 而tensor.eq()是每个元素比较是否相等
          coils_61.append(use_coil_dic[key][index,:])

        elif coil_curve.equal(self.curve_99):
          coils_99.append(use_coil_dic[key][index,:])

      mix_coils_list = []
      num_61 = len(coils_61)
      num_99 = len(coils_99)
      r_61 = 0
      r_99 = 0
      r_mix = 0
      if (num_61 % 3) == 0 and num_61 >= 3:  # 取余等于0，说明板卷数正好为3的倍数

        r_61 = self.cal_reward_(coils_61, curve_61_flag=True, mixMatchFlag=False)

      elif (num_61 % 3) != 0 and num_61 > 3 :  # 取余不等于0且元素个数大于3，说明板卷数为4 5 7 8 ...

        divide = (num_61 // 3) * 3
        r_61 = self.cal_reward_(coils_61[0:divide], curve_61_flag=True, mixMatchFlag=False)
        mix_coils_list.extend(coils_61[divide:])
      elif num_61 >0 and num_61 < 3:
        mix_coils_list.extend(coils_61)


      if (num_99 % 3) == 0 and num_61 >= 3:  # 取余等于0，说明板卷数正好为3的倍数

        r_99 = self.cal_reward_(coils_99, curve_61_flag=False, mixMatchFlag=False)

      elif (num_99 % 3) != 0 and num_99 > 3 :  # 取余不等于0且元素个数大于3，说明板卷数为4 5 7 8 ...

        divide = (num_99 // 3) * 3
        r_99 = self.cal_reward_(coils_99[0:divide], curve_61_flag=False, mixMatchFlag=False)
        mix_coils_list.extend(coils_99[divide:])

      elif num_99 >0 and num_99 < 3:

        mix_coils_list.extend(coils_99)
        # print(mix_coils_list)

      r_mix = self.cal_reward_(mix_coils_list, mixMatchFlag=True)
      # 计算Reward
      r_total = r_61 + r_99 + r_mix



      R.append(r_total)


    if len(R):  # 非空
      reward = torch.tensor(R, dtype=torch.float32)
    else:  # 空list
      reward = torch.tensor(-20, dtype=torch.float32)
    '''
    return unuse_coil_dic, use_coil_dic

  def forward(self, date, input, encoder_output, encoder_hidden, memory_lengths=None):
    '''

    :param date: int,板卷倒料时间
    注意seq_len和max_in_seq_len的区别，seq_len为每个句子的实际最大长度，max_in_seq_len为padding后的维度（人为指定）
    :param input(torch.float32): [max_seq_len, batch_size, input_size]=[21, 3, 7]
    :param encoder_output(torch.float32): [seq_len, batch_size, hidden_size]，此seq_len为同一batch中的句子实际最大长度
    :param encoder_hidden: = (h_t,c_t)  h_t.shape=c_t.shape=[layer_num, batch_size, hidden_size]=[1, batch_size, 128]
    h_t, c_t = torch.float32
    :param memory_lengths(torch.int32): [batch_size, 1]  存放的是input中每个句子真正长度seq_len
    :return:
    '''

    batch_size = encoder_output.size(1)
    decoder_seq_len = encoder_output.size(0)  # 经过pack_padded_sequence操作后，decoder_seq_len变成了每个batch的实际最大个数
    hidden = encoder_hidden[0]   # h_t
    context = encoder_hidden[1]  # c_t
    # Decoding states initialization
    # 扩展decoder_input的维度，与input的维度保持一致
    # decoder_input = to_cuda(torch.rand(batch_size, self.input_size)) #初始化decoder_input输入，这个地方有点疑问？参数需要初始化 而且需要学习

    decoder_input = nn.Parameter(torch.FloatTensor(self.input_size))
    decoder_input.data.uniform_(-(1. / math.sqrt(self.input_size)), 1. / math.sqrt(self.input_size))
    # [batch_size, input_size]，使用repeat()并在新扩充的维度上赋值为与原来一样的值
    decoder_input = decoder_input.unsqueeze(0).repeat(batch_size, 1)
    #decoder_input = decoder_input.unsqueeze(0).repeat(batch_size, 1).cuda()

    prev_probs = []
    prev_idxs = []
    mask = torch.zeros(batch_size, decoder_seq_len).byte()

    # if self.use_cuda:
    #   mask = mask.cuda()

    idxs = None

    for i in range(decoder_seq_len):
      # decoder_input.unsqueeze(0).cuda()
      # print(decoder_input.unsqueeze(0))
      # print(hidden)
      # print(context)
      # hidden.shape = context.shape = [layer_num, batch_size, hidden_size]
      # decoder_input.unsqueeze(0).shape = [1,batch_size, input_size]
      decoder_output, (hidden, context) = self.rnn(decoder_input.unsqueeze(0), (hidden, context))

      # encoder_output.shape = [seq_len, batch_size, hidden_size]
      # hidden.shape = [layer_num, batch_size, hidden_size]
      # memory_lengths.shape = [batch_size, 1]
      align_score = self.attention(encoder_output, hidden, memory_lengths)


      # align_score.shape = mask.shape = [batch_size, input_size]
      # logits.shape = [batch_size, input_size]
      # probs = [batch_size, input_size]
      logits, mask = self.apply_mask_to_logits(align_score, mask, idxs)
      probs = F.softmax(logits, 1)  # dim=1,即保留列，按照一行所有的数应用softmax进行计算


      probs_clone = probs.clone()


      # multinomial(1)：根据softmax之后的logits进行随机采样（1表示随机采样个数）,注意这里采样返回的是index
      # multinomial接收input：input张量可以看成一个权重张量，每一个元素代表其在该行中的权重。
      # 如果有元素为0，那么在其他不为0的元素被取干净之前，这个元素是不会被取到的。
      idxs = probs_clone.multinomial(1).squeeze(1)

      # 如果设置-inf的数字依然被采样到，那么进行重新采样
      # 可以考虑注释掉此段代码，因为短句子的start_token一定会被重复采样，觉得可能有些影响效率
      # for old_idxs in prev_idxs:
      #   if old_idxs.eq(idxs).data.any().bool():  # 是否全部为 False，则返回 False，如果有一个为 True，则返回 True。
      #     print('RESAMPLE!')
      #     idxs = probs_clone.multinomial(1).squeeze(1)
      #     break

      # 此句代码有必要留着： 此处可以添加合同信息喂给decoder
      decoder_input = input[idxs.data, [i for i in range(batch_size)], :]

      # probs.shape = [batch, input_size]
      # idxs.shape = [idxs, input_size]
      prev_probs.append(probs_clone)
      prev_idxs.append(idxs)

    probs_clone = prev_probs
    action_idxs = prev_idxs

    # print(probs_clone)
    # print(action_idxs)

    actions = []  # 存储每个动作,即每次挑出来的板卷信息
    for action_id in action_idxs:
      # action_id.data： 板卷的index
      actions.append(input[action_id.data, [x for x in range(batch_size)],:])

    action_probs = []  # 存储每个动作数字被选的概率，最后一个为1
    # [sequence, batch_size]
    for prob, action_id in zip(probs_clone, action_idxs):
      action_probs.append(prob[[x for x in range(batch_size)], action_id.data])


    #input.shape = [max_in_seq_len, batch_size, input_size]
    unuse_coil_dic, use_coil_dic = self.reward(date, input, action_idxs, actions)

    # print(unuse_coil_dic)
    # action_probs 每个动作被选择的概率
    # action_idxs 每个动作的index
    # actions 每个动作的具体信息（动作本身）
    return action_probs, actions, action_idxs, unuse_coil_dic, use_coil_dic


