# -*- coding: utf-8 -*-
from lib2to3.pgen2.tokenize import tokenize
from sre_parse import Tokenizer
import torch 
import numpy as np
import random

# 固定随机数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# input = torch.distributions.exponential.Exponential(rate=1).sample([200000]).cuda()
# input = np.random.exponential(scale=1,size=(10))
# data = np.around(input, 2).astype(np.float32)  # 保留两位小数

setup_seed(20)
samples = torch.distributions.exponential.Exponential(rate=1).sample([10]).cuda()
index = np.arange(2, 12)
    
vocab = dict(zip(index, samples))
vocab[0] = '<bos>'
vocab[1] = '<eos>'
print(vocab)

# print(samples)