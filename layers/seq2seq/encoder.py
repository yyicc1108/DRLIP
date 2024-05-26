#!/usr/bin/env Python
# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

def rnn_factory(rnn_type, **kwargs):
  pack_padded_seq = True
  if rnn_type in ["LSTM", "GRU", "RNN"]:
    rnn = getattr(nn, rnn_type)(**kwargs)
    print(rnn)
  return rnn, pack_padded_seq

class EncoderBase(nn.Module):
  """ encoder base class
  """
  def __init__(self):
    super(EncoderBase, self).__init__()

  def forward(self, src, lengths=None, hidden=None):
    """
    Args:
      src (FloatTensor) : input sequence 
      lengths (LongTensor) : lengths of input sequence
      hidden : init hidden state
    """
    raise NotImplementedError()



class RNNEncoder(EncoderBase):
  """ RNN encoder class

  Args:
    rnn_type : rnn cell type, ["LSTM", "GRU", "RNN"]   ="LSTM"
    bidirectional : whether use bidirectional rnn      =True
    num_layers : number of layers in stacked rnn       =1
    input_size : input dimension size                  =4
    hidden_size : rnn hidden dimension size            =128
    dropout : dropout rate                             =0.0
    use_bridge : TODO: implement bridge
  """

  def __init__(self, rnn_type, bidirectional, num_layers, input_size, hidden_size, dropout):
    super(RNNEncoder, self).__init__()
    if bidirectional:
      assert hidden_size % 2 == 0
      hidden_size = hidden_size // 2

    self.rnn, self.pack_padded_seq = rnn_factory(rnn_type,
      input_size=input_size,
      hidden_size=hidden_size,
      bidirectional=bidirectional,
      num_layers=num_layers,
      dropout=dropout)

  def forward(self, src, lengths=None, hidden=None):
    """
      Args:
    src (torch.float32) : [max_in_seq_len, batch_size, input_size] = [max_in_seq_len, batch_size, 7]
    lengths (torch.int32) : lengths of input sequence  shape = [batch_size, 1]
    hidden : init hidden state

    此forward中使用了from torch.nn.utils.rnn import pack_padded_sequence as pack方法，此方法特别关键。
    在参数设置时，max_in_seq_len被设置成1000长度，因此绝大部分时候这个长度都是过剩的且都是用0来进行填补的，所以就造成非常多无用的字符通过了LSTM
    此时相当于噪声，为此需要LSTM去处理变长输入的需求。 具体可以参考https://zhuanlan.zhihu.com/p/34418001
    """

    if self.pack_padded_seq and lengths is not None:

      lengths = lengths.view(-1).tolist()  # pack_padded_sequence 要求类型为list(或一维tensor向量，int64类型)

      packed_src = pack(src, lengths, enforce_sorted=False) #enforce_sorted为False，Pytorch会自动给src做排序

      encoder_output, hidden_final = self.rnn(packed_src, hidden)      # hidden = None 可以传入，不起作用

      # unpack(encoder_output)返回一个tuple，包含被填充后的序列，和batch中序列的长度列表
      encoder_output = unpack(encoder_output)[0]

    # RNN(encoder计算过程)：
    # encoder_output.shape = [seq_len, batch_size, input_size] * [input_size, hidden_size] = [seq_len, batch_size, hidden_size]
    # hidden_final = (h_t, c_t), h_t.shape = c_t.shape = [layer_num, batch_size, hidden_size]
    # 注意：经过pack和unpack操作后，seq_len已经不是原来的max_in_seq_len，而是为同一batch中的句子实际最大长度

    return encoder_output, hidden_final