import torch
import numpy as np
import case1_loss

class Batch:
    """Object for holding a batch of data with mask during training.训练期间用于保存一批带mask的数据的对象"""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src  # 与EncoderDecoder中forward函数的src一致
        self.src_mask = (src != pad).unsqueeze(-2) # 将src中pad部分掩盖，使维度与scores保持一致
        if tgt is not None: # tgt即目标句子
            self.tgt = tgt[:, :-1] # 需要去掉最后一个词，tgt存储的是decoder的输入，所以不会出现最后一个词，即`<bos> 我 爱 你`（没有'<eos>'）
            self.tgt_y = tgt[:, 1:] # tgt_y存储希望预测的结果，即decoder的预测输出结果，去掉第一个词'<bos>'，即“我 爱 你 <eos>”
            # self.tgt_mask = self.make_std_mask(self.tgt, pad) # tgt_mask & subsequent_mask
            self.ntokens = (self.tgt_y != pad).data.sum() # 该batch的tgt_y的总token的数量，去除'pad'部分的词数，同时也去掉了前面的'<bos>'

    # @staticmethod
    # def make_std_mask(tgt, pad): # 生成tgt_mask
    #     "Create a mask to hide padding and future words."
    #     tgt_mask = (tgt != pad).unsqueeze(-2) # 生成非句子成分的mask，加入pad
    #     tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as( 
    #         tgt_mask.data
    #     )
    #     return tgt_mask

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def data_gen(V, batch_size, nbatches):
    """
    生成一组随机数据。（该方法仅用于Demo）
    :param V: 词典的大小
    :param batch_size
    :param nbatches: 生成多少个batch
    :return: yield一个Batch对象
    """

    # 生成{nbatches}个batch
    for i in range(nbatches):
        # 生成一组输入数据
        data = torch.randint(1, V, size=(batch_size, 3))
        # 将每行的第一个词都改为1，即"<bos>"
        data[:, 0] = 1
        # 该数据不需要梯度下降
        src = data.requires_grad_(False).clone().detach()
        
        true_data = data[:, 1:] # 去掉第一列
        value = case1_loss.index_to_value(true_data)
        value = torch.tensor(np.round(np.array(value), 4)) # value是src对应的样本值
        # print(value, value.shape)
        print(true_data)
        # g1 = value[:,0]
        # g2 = value[:,1]
        # get_keys(case1_loss.readvocab, )
        # g1 * g2
        # print(value, value[:,0])
        
        tgt = data.requires_grad_(False).clone().detach()
        # 返回一个Batch对象
        # print("src", src, src.shape)
        # print("tgt", tgt, tgt.shape)
        yield Batch(src, tgt, 0)
        
result = data_gen(V=1000, batch_size=80, nbatches=2)
for i in result:
    print(i)