from audioop import mul
import random
import numpy as np
from sklearn.utils import resample
import torch
from decimal import *

# 固定随机数
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# 量化+解量化
# def Quantization(SigmoidData):
#     # Quantization and De-quantization
#     # N = 2
#     result = Decimal('SigmoidData').quantize(Decimal('0.000'))
#     # stepsize = (10-0)/10000
#     # # Encode
#     # Migmoid_quant_rise_ind = np.floor(SigmoidData / stepsize)
#     # # Decode
#     # Migmoid_quant_rise_ind = Migmoid_quant_rise_ind * stepsize
#     # return Migmoid_quant_rise_ind 
#     return result

# 创建长度为1000的字典vocab
# def getData():
#     # setup_seed(20)
#     # samples = torch.distributions.exponential.Exponential(rate=1).sample([998])   # 共生成998个单词，还有2个是bos、eos
#     sample = np.random.exponential(scale=1.0, size=[998])
#     sample = np.round(sample, decimals=3)
#     # print(sample)

#     index = np.arange(2, 1000)    # 下标 = 单词数 + 1
#     # print("输出的下标",DataIndex)

#     vocab = dict(zip(index, sample))
#     vocab[0] = 0
#     vocab[1] = 1
#     torch.save(vocab, 'vocab_np')    # 将字典存入vocab文件
#     # print("字典",vocab)
#     return vocab    # len(vocab) = 1000，共有998个单词

# def getData():
#     # setup_seed(20)
#     # samples = torch.distributions.exponential.Exponential(rate=1).sample([998])   # 共生成998个单词，还有2个是bos、eos
#     sample = np.random.exponential(scale=1.0, size=[1000]).astype(np.float32)
#     sample = np.round(sample, decimals=3)
#     # print(sample)

#     # index = np.arange(2, 1000)    # 下标 = 单词数 + 1
#     # print("输出的下标",DataIndex)

#     # vocab = dict(zip(index, sample))
#     # vocab[0] = 0
#     # vocab[1] = 1
#     torch.save(sample, 'vocab_matrix')    # 将字典存入vocab文件
#     # print("字典",vocab)
#     return sample    # len(vocab) = 1000，共有998个单词

# data = getData()
# readvocab = torch.load('vocab_matrix')     # 读取vocab文件
# vocab = np.array(readvocab.values())

# 创建包含1000个随机数的字典
readvocab = np.random.exponential(scale=1.0, size=[1000])
# vocab:torch.Size([1000]) vocab_multi:torch.Size([1000, 1000]) vocab_square:torch.Size([1000, 1000])
vocab = torch.from_numpy(readvocab)     # 下标是[0,999]
# g1*g2的单词表，注意reshape(1,-1)后是二维张量
vocab_multi = torch.mm(vocab.reshape(1000, 1), vocab.reshape(1, 1000)).reshape(1,-1)
# 0.5*g1^2*g2^2的单词表
vocab_square = torch.mm(vocab.reshape(1000, 1)**2, vocab.reshape(1, 1000)**2).reshape(1,-1)/2
# print(vocab)
# 如果输入的index是[3,2]，则multi对应的index是[0,2*1000+2]
# print(vocab[2],vocab[1],vocab_multi[0,2001],vocab_square[0,2001])

# a = torch.range(1,4)
# b = a
# mm = torch.mm(a.reshape(4,1), b.reshape(1,4)).reshape(1,-1)
# # print(a, b, mm)
# print(a[2], b[1], mm[0,9])


index1 = torch.tensor([[1, 2], [3, 4]])
index2 = torch.tensor([[1, 2], [3, 4]])
# print(torch.mm(index1.reshape(4, 1)**2, index2.reshape(1, 4)**2)/2)

# value = []
# g1 = [i[0] for i in value]  
# g2 = [i[1] for i in value]

# 将输入的下标转换为字典中检索的样本值
def index_to_value(indexes):
    # vocab_data = getData()
    value = []
    for index in indexes:
        input1 = index[0].item()
        input2 = index[1].item()
        
        result = []
        result.append(readvocab[input1])
        result.append(readvocab[input2])
        value.append(result)
    return value    # 二维list，里面的值是tensor

# a = index_to_value(index1)
# print(a, readvocab[2], readvocab[3])
# print(readvocab[2], readvocab[3])

def value_to_g(value):
    g1 = np.zeros((0,))
    g2 = np.zeros((0,))
    # g1 = [i[0] for i in value]
    # g2 = [i[1] for i in value]
    for i in value:
        # print(i)
        g1 = np.append(g1, i[0])
        # print(g1)
    for i in value:
        g2 = np.append(g2, i[1])
    # g1 = np.append(i[0] for i in value)
    # g2 = np.append(i[1] for i in value)
    # print(g1, type(g1), g2, type(g2))
    # g1 = torch.tensor(np.array(g1))
    g1 = torch.tensor(g1).cuda()
    g1.requires_grad = True
    # g2 = torch.tensor(np.array(g2))
    g2 = torch.tensor(g2).cuda()
    g2.requires_grad = True
    # print(g1, type(g1), g2, type(g2))
    return g1, g2   # 二维tensor


def case1_loss1(g1, g2, g1_hat, g2_hat):
    # print("损失函数",g1,g1_hat)
    eps = 1e-8 
    loss1 = torch.sqrt(torch.pow((g1*g2 - g1_hat*g2_hat),2) + torch.pow((g1**2 * g2**2)/2 - (g1_hat**2 * g2_hat**2)/2, 2) + eps).cuda()

    loss1 = torch.sum(loss1).cuda()

    return loss1.cuda()

# 面向目标
def case1_loss2(g1, g2, g1_hat, g2_hat):
    h1 = 2*g1*g2 - g1**2*g2**2/2
    h2 = g1**2*g2**2 - g1*g2

    loss2_1 = 3*(g1*g2 - g1**2*g2**2/2)**2
    loss2_2 = (g1_hat*g2_hat - h1)**2
    loss2_3 = (g1_hat**2*g2_hat**2/2 - h2)**2
    loss2_4 = (g1_hat*g2_hat - g1_hat**2*g2_hat**2/2)**2

    loss2 = (loss2_1 - loss2_2 - loss2_3 - loss2_4)**2

    loss2 = torch.sum(loss2).cuda()

    return loss2.cuda()

# result = index_to_value(index1)
# print(result, readvocab[2], readvocab[3])
# g1, g2 = value_to_g(index_to_value(index1))
# g1_head, g2_head = value_to_g(index_to_value(index1))
# result = case1_loss1(g1, g2, g1_head, g2_head)
# print(g1, type(g1), g2, type(g2), result, type(result))
# print(g1, type(g1), g2, type(g2)) 


# g1, g2 = value_to_g(index_to_value(index1))
# a, b = value_to_g(index_to_value(index2))
# loss = case1_loss1(g1, g2, a, b)
# loss.requires_grad_(True)  #这里应该是因为如果将最后一层的模型参数梯度关闭，则计算出来的loss也没有梯度，不能追踪，所以要将loss的梯度设置为True
# loss.backward()
# print(loss, type(loss), loss.requires_grad)