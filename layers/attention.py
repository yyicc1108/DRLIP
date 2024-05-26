#!/usr/bin/env Python
# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F

def sequence_mask(lengths, max_len=None):
  """ Crete mask for lengths
  Args:
    lengths (LongTensor) : lengths (batch_size)
    max_len (int) : maximum length
  Return:
    mask (batch_size, max_len)
  """
  batch_size = lengths.numel()  # batch_size = 3 numel() 返回容器内元素个数
  max_len = max_len or lengths.max()  #lengths.max()=6 输入的最大维数

  # print(torch.arange(0, max_len))
  # print(torch.arange(0, max_len).type_as(lengths))
  # print(torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1))
  # print(lengths)
  # print(torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths))

  return (torch.arange(0, max_len)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths))

class Attention(nn.Module):
  """ Attention layer
  Args:
    attn_type : attention type ["dot", "general"]
    hidden_size : input hidden_sizeension size
  """
  def __init__(self, attn_type, hidden_size):
    super(Attention, self).__init__()

    self.attn_type = attn_type

    bias_out = attn_type == "mlp"
    self.linear_out = nn.Linear(hidden_size *2, hidden_size, bias_out)
    if self.attn_type == "general":
      self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
    elif self.attn_type == "dot":
      pass
    else:
      raise NotImplementedError()
  
  def score(self, src, tgt):
    """ Attention score calculation
    Args:
      src : source values (batch_size, src_seq_len, hidden_size)    encoder_output
      tgt : target values (batch_size, tgt_seq_len, hidden_size)    decoder_output
    """
    batch_size, src_seq_len, hidden_size = src.size()
    _, tgt_seq_len, _ = tgt.size()

    if self.attn_type in ["genenral", "dot"]:
      tgt_ = tgt
      if self.attn_type == "general":
        tgt_ = self.linear(tgt_)
      src_ = src.transpose(1, 2)
      return torch.bmm(tgt_, src_)   #两个tensor的矩阵乘法
    else:
      raise NotImplementedError()
  
  def forward(self, src, tgt, src_seq_lengths=None):
    """
    Args:
      src : source values (batch_size, src_seq_len, hidden_size)  encoder_output
      tgt : target values (batch_size, tgt_seq_len, hidden_size)  decoder_output
      src_seq_lengths : source values length (batch_size, 1)      inp_length
    """

    if tgt.dim() == 2:  # tgt.dim() = 3
      one_step = True
      src = src.unsqueeze(1)
    else:
      one_step = False
    
    batch_size, src_seq_len, hidden_size = src.size()
    _, tgt_seq_len, _ = tgt.size()

    align_score = self.score(src, tgt)  #align_score.shape=[batch_size, tgt_seq_len, src_seq_len]=[batch_size,7,6]

    # print(align_score.shape)

    if src_seq_lengths is not None:
      mask = sequence_mask(src_seq_lengths)
      # (batch_size, max_len) -> (batch_size, 1, max_len)
      # so mask can broadcast
      mask = mask.unsqueeze(1)
      align_score.data.masked_fill_(~mask, -float('inf')) # 利用了broadcast机制
      # print(~mask)
      # print(align_score)
    
    # Normalize weights
    align_score = F.softmax(align_score, -1)
    # print(align_score)

    c = torch.bmm(align_score, src)

    concat_c = torch.cat([c, tgt], -1)
    attn_h = self.linear_out(concat_c)

    if one_step:
      attn_h = attn_h.squeeze(1)
      align_score = align_score.squeeze(1)
    
    return attn_h, align_score