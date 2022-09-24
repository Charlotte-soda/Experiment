# -*- coding: utf-8 -*-
import random
from ast import In
from lib2to3.pgen2.tokenize import tokenize
from sre_parse import Tokenizer

import numpy as np
import torch


# 固定随机数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# input = torch.distributions.exponential.Exponential(rate=1).sample([200000])
# input = np.random.exponential(scale=1,size=(10))
# data = np.around(input, 2).astype(np.float32)  # 保留两位小数

def getData(DataIndex):
    setup_seed(20)
    # samples = torch.distributions.exponential.Exponential(rate=1).sample([10])
    samples = torch.distributions.exponential.Exponential(rate=1).sample([1000])
    # print(samples)

    index = np.arange(2, 1000 - 1)
    # print("输出的下标",DataIndex)

    vocab = dict(zip(index, samples))
    vocab[0] = '<bos>'
    vocab[1] = '<eos>'
    # print("字典",vocab)

    # 获取下标对应的数
    for index in DataIndex:
        for id in index:
            print(vocab[id])

    # print(samples)

# if __name__ == "__main__":
#     # 传入下标
#     A = torch.arange(10, dtype=torch.float32).reshape((1,10))
#     B = A.detach().numpy()  # tensor转换为ndarray
#     print(A)
#     print(B)
#     getData(B)

