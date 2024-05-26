#!/usr/bin/env Python
# coding=utf-8

import numpy as np
import torch
from torch.utils.data import Dataset
from copy import copy


# 要利用pytorch中的DataLoader，首先需要创建自己的数据集对象
# 创建的类里面至少包含3个函数：
# __init__：传入数据，或者像下面一样直接在函数里加载数据
# __len__： 返回这个数据集一共有多少个item
# __getitem__：返回一条训练数据，并将其转换成tensor

class CHDataset(Dataset):
    """ Dataset for Convex Hull Problem data
    Args:
      filename : the dataset file name
      max_in_seq_len :  maximum input sequence length
      max_out_seq_len : maximum output sequence length
    """

    def __init__(self, filename, max_in_seq_len, max_out_seq_len):
        super(CHDataset, self).__init__()
        self.max_in_seq_len = max_in_seq_len
        self.max_out_seq_len = max_out_seq_len
        self.START = [0, 0, 0]
        self.END = [0, 0, 0]
        self.input_size = 3
        self.slab_ord_data = list()
        self._load_data(filename[0], 2)
        self._load_data(filename[1], 3)
        self.length = min(len(self.slab_ord_data[0]), len(self.slab_ord_data[1]))
        print(len(self.slab_ord_data[0]), len(self.slab_ord_data[1]))
        # print(self.slab_ord_data[0][0])
        # print(self.slab_ord_data[1][0])


    def _load_data(self, filename, input_size_):
        '''

        输入数据start token必须用[0,0]，输出数据的start token必须用[0,0]

        输出或输入数据的维度不足最大维度，则用[0,0]在末尾补齐

        outp_out必须先用[0,0]填充？ end_token

        '''
        self.input_size = input_size_
        self.START = [0] * input_size_
        self.END = [0] * input_size_
        with open(filename, 'r') as f:
            data = []
            for line in f:

                inp = line.strip()    # strip()：去除首尾空格  返回的是字符串inp outp
                # print(inp, type(inp))

                #使用map，将inp中的
                inp = list(map(float, inp.strip().split())) # 用map将分割好的字符串变成float,最后变成List

                # Padding input
                inp_len = len(inp) // self.input_size      # //表示整数除法 inp_len=3

                # add start_token to inp
                inp = self.START + inp
                inp_len += 1

                # 小于max_in_seq_len的用[0,0,0]进行padding补齐
                assert self.max_in_seq_len + 1 >= inp_len
                for i in range(self.max_in_seq_len + 1 - inp_len):
                    inp += self.END

                # print(inp)
                inp = np.array(inp).reshape([-1, self.input_size])  # inp.shape=(max_in_seq_len, input_size) = (1000, 3)
                inp_len = np.array([inp_len])         # 每个句子的真正长度

                # print(type(inp.astype("float32")), type(inp_len))
                data.append((inp.astype("float32"), inp_len))  # data为list类型，存入的是元祖（max_in_seq_len，真正句子长度inp_len）
                # print(data, type(data))

            self.slab_ord_data.append(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inp1, inp_len1 = self.slab_ord_data[0][index]
        inp2, inp_len2 = self.slab_ord_data[1][index]
        return inp1, inp_len1, inp2, inp_len2
