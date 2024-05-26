#!/usr/bin/env Python
# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F


def sequence_mask(lengths, max_len=None):
    """ Create mask for lengths
    Args:
      lengths (torch.int32) : lengths      [batch_size, 1]
      max_len (int) : maximum length
    Return:
      mask (batch_size, max_len)
    """

    batch_size = lengths.shape[0]

    max_len = max_len or lengths.max()  # lengths中最大的长度

    # print(torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths))
    # 比较函数：torch.ge(), torch.gt(), torch.le(), torch.lt(), 详见https://blog.csdn.net/weixin_40522801/article/details/106579068
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths))

class Attention1(nn.Module):
    """ Attention layer
    Args:
      attn_type : attention type ["dot", "general"]
      hidden_size : input hidden_sizeension size
    """

    def __init__(self, attn_type, hidden_size):
        super(Attention1, self).__init__()

        self.attn_type = attn_type
        self.hidden_size = hidden_size
        self.weight_size = 128

        if self.attn_type == 'Bahdanau':
            self.W1 = nn.Linear(self.hidden_size, self.weight_size, bias=False)   # nn.Linear线性层不要偏置项 bias
            self.W2 = nn.Linear(self.hidden_size, self.weight_size, bias=False)
            self.vt = nn.Linear(self.weight_size, 1, bias=False)
        elif self.attn_type == 'dot':
            raise NotImplementedError()

    def score(self, encoder_output, hidden):
        """ Attention score calculation
        Args:
          encoder_output.shape = [seq_len, batch_size, hidden_size]
          hidden.shape =  [layer_num, batch_size, hidden_size]
        """
        if self.attn_type == 'Bahdanau':
            # encoder_output * W1 = [seq_len, batch_size, hidden_size]*[hidden_size, weight_size]=[seq_len, batch_size, weight_size]
            # hidden.squeeze(0) * W2 = [batch_size, hidden_size] * [hidden_size, weight_size] = [batch_size, weight_size]
            sum = torch.tanh(self.W1(encoder_output) + self.W2(hidden.squeeze(0))) # 利用了广播机制
            logits = self.vt(sum).squeeze()    # [seq_len, batch_size, weight_size]*[weight_size, 1]=[seq_len,batch_size]

        elif self.attn_type == 'dot':
            # 需要维度的变换
            # hidden.shape = [batch_size x hidden_size]
            # encoder_output = [batch_size x seq_len x hidden_size]  矩阵乘法时，batch_size放到前面，可忽略
            hidden = hidden.unsqueeze(0)
            encoder_output = encoder_output.transpose(0, 1)
            logits = torch.bmm(encoder_output, hidden).squeeze(2)  # [seq_len, batch_size,1]

        return logits

    def forward(self, encoder_output, hidden, src_seq_lengths=None):
        """
        Args:
          encoder_output.shape = [seq_len, batch_size, hidden_size]
          hidden.shape =  [layer_num, batch_size, hidden_size]
          src_seq_lengths : source values length [batch_size, 1]  存放的是input中每个句子真正长度seq_len
        """
        # align_score.shape = [seq_lne,batch_size]
        align_score = self.score(encoder_output, hidden).view(-1, 32)
        # align_score.shape = [batch_size, seq_len]
        # print(align_score)
        align_score = align_score.transpose(0, 1)


        if src_seq_lengths is not None:
            mask = sequence_mask(src_seq_lengths)
            # masked_fill_(mask, value): 用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致
            align_score.data.masked_fill_(~mask, -float('inf'))  # 利用了broadcast机制

        return align_score