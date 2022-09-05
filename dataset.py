# -*- coding: utf-8 -*-
from lib2to3.pgen2.tokenize import tokenize
from sre_parse import Tokenizer
from matplotlib.style import library
import torch 
import numpy as np
import collections
import re
from d2l import torch as d2l

"""
文本预处理
1、读取文本
2、分词
3、建立词典 将每个词映射到一个唯一的索引
4、将文本从词的序列转换为索引的序列,作为模型的输入
"""

# input = torch.distributions.exponential.Exponential(rate=1).sample([200000]).cuda()
input = np.random.exponential(scale=1,size=(10))
data = np.around(input, 2).astype(np.float32)  # 保留两位小数

a = [str(n).split() for n in data]

def process():
    return [re.sub('^[0-9]+', ' ', str(data)).strip()]

input = process()
# print(data, '\n', str(data), '\n', input)
print(a)