import random
import numpy as np
import torch
from decimal import *

# 固定随机数
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

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

# # 创建长度为1000的字典vocab
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
readvocab = torch.load('vocab_matrix')     # 读取vocab文件
# vocab = np.array(readvocab.values())
vocab = torch.from_numpy(readvocab)
# vocab = torch.tensor(vocab)
# print(vocab, vocab.shape)



index1 = torch.tensor([[2, 3], [4, 5], [6, 7], [8, 9]])
index2 = torch.tensor([[50, 3], [46, 5], [26, 7], [5, 34]])

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


def case1_loss1(input1, input2, out1, out2):
    # print("损失函数",input1,out1)
    eps = 1e-8 
    loss1 = torch.sqrt(torch.pow((input1*input2 - out1*out2),2) + torch.pow((input1**2 * input2**2)/2 - (out1**2 * out2**2)/2, 2) + eps).cuda()
    # loss1.requires_grad = True
    # print(loss1, loss1.shape)  # loss1:[80]
    loss1 = torch.sum(loss1).cuda()
    # loss1.requires_grad = True
    # print("和",loss1)
    
    return loss1.cuda()

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

