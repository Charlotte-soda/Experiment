import copy
import math
import os
import sys
import time
from tkinter import image_names
import warnings  # 用于忽略警告日志
from os.path import exists

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter 

from DataProcess.data import getData


"""
batch_size:即一次训练所抓取的数据样本数量； batch_size的大小影响训练速度和模型优化，也影响每一epoch训练模型次数。即有多少个句子。
length：即每个句子有多少个单词。
d_model：嵌入维度，一般设为512维。
"""

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore") # 设置忽略警告
RUN_EXAMPLES = True


batch_size = 80
V = 1001

def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)

# 验证时不进行参数更新
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

# 验证时不进行学习率调整
class DummyScheduler:
    def step(self):
        None
    
# 量化器
def Quantization(SigmoidData):
    # Quantization and De-quantization
    N = 4
    stepsize = (1.0-(0))/(2**N)
    # Encode
    Migmoid_quant_rise_ind = torch.floor(SigmoidData / stepsize)
    # Decode
    Migmoid_quant_rise_ind = Migmoid_quant_rise_ind * stepsize + stepsize / 2
    
    return Migmoid_quant_rise_ind    
        
# 标准的模型架构，返回值为decoder输出
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, d_model):
        super(EncoderDecoder, self).__init__()
        # 原模型
        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()
        self.src_embed = src_embed.cuda()
        self.tgt_embed = tgt_embed.cuda()
        self.generator = generator.cuda()

        # 自定义部分
        self.linear1 = nn.Linear(d_model, 510).cuda()
        self.linear2 = nn.Linear(510,d_model).cuda()
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        # return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        x = self.encode(src, src_mask).cuda()
        
        x = self.linear1(x) # 降维
        x = self.sigmoid(x).cuda()
        x = Quantization(x).cuda()
        output = self.linear2(x)

        return self.decode(output, src_mask, tgt, tgt_mask).cuda()


    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask).cuda()

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask).cuda()
    
# 将decoder结果输入到Linear + Softmax，预测下一个token
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab): # vocab是词典的大小
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1) # log_softmax替换softmax，加快运算速度，提高数据稳定性

# Encoder和Decoder堆叠
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        """
        初始化传入两个参数：
        layer：要堆叠的层，即EncoderLayer类
        N：堆叠多少次
        """
        self.layers = clones(layer, N).cuda()
        self.norm = LayerNorm(layer.size).cuda()   # 对应 Add & Norm 中的 Norm 部分

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        
        """
        input: [batch_size, 每个句子的词数, d_model] 
        mask: 即src_mask，将非句子真实内容的pad部分进行mask
        """
        for layer in self.layers:   # 一层一层执行，将前一个Encoderlayer的输出作为下一层的输入
            x = layer(x, mask)
        return self.norm(x)
    
# LayerNorm的具体实现
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6): # features即特征数d_model，int类型
        super(LayerNorm, self).__init__()
        # a_2对应gamma(γ=1)，b_2对应beta(β=0)
        # nn.Parameter：将两个参数作为模型参数，之后进行梯度下降
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  # 一个很小的数，防止分母为0

    def forward(self, x):   # x为attention层或feed forward层的输出，shape同encoer输入
        # x: [batch_size, 词数, d_model]
        # layernorm取同一个样本的不同特征进行归一化
        mean = x.mean(-1, keepdim=True) # 按最后一个维度d_model求均值，并保持shape不变
        std = x.std(-1, keepdim=True)   # 同上，求方差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 # 按layernorm公式计算，进行归一化
    
# Add & Norm 的整体部分，add和norm两个子层之间的残差连接        
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):  # size即d_model，即词向量维度
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # x：本层的输入，即前一层的输出
        # sublayer：指attention层或者feed forward层，两者的计算方式相同
        return x + self.dropout(sublayer(self.norm(x))) # 为了简化代码，修改实现顺序
        # return self.norm(x + self.dropout(sublayer(x))) # 原模型顺序

# 单个EncoderLayer的具体实现：attention层 + feed forward层        
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout): 
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn.cuda()
        self.feed_forward = feed_forward.cuda()
        # 克隆两个残差连接，分别给attention层和feed forward层使用
        self.sublayer = clones(SublayerConnection(size, dropout), 2).cuda()
        self.size = size    # size即d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # sublayer[0]即attention层的计算
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # sublayer[1]即feed forward层的计算
        return self.sublayer[1](x, self.feed_forward).cuda()
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N).cuda()
        self.norm = LayerNorm(layer.size).cuda()

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        memory：即encoder最后一层的输出
        src_mask：即mask掉输入中非句子的pad部分
        tgt_mask：即在训练时mask掉某时刻该单词之后的单词，模拟inference
        
        x:通过decoder中嵌入层和位置编码后的“输入即模型图中的outputs”，x:[batch_size, 词数, d_model]
          预测时，x的词数会不断变化，第一次为(1,1,128)，第二次为(1,2,128)....第7次为(1,7,128)
        """
        
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask).cuda()
        return self.norm(x).cuda()

# 单个DecoderLayer的具体实现：attention层 + feed forward层 + cross-attention层
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size    # size即d_model，词向量的维度
        self.self_attn = self_attn.cuda()
        self.src_attn = src_attn.cuda()
        self.feed_forward = feed_forward.cuda()
        # 克隆三个残差连接，分别给masked attention层、cross-attention层和feed forward层使用
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        # sublayer[0]指masked attention层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)).cuda()
        # sublayer[1]指cross-attention层，其中key和value是encoder的输出即memory
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)).cuda()
        # sublayer[2]指feed forward层
        return self.sublayer[2](x, self.feed_forward).cuda()
    
# encoder不注意非句子的pad部分。decoder不注意后面的词，同时也不注意pad部分  
def subsequent_mask(size):
    "Mask out subsequent positions."
    # 生成一个大小为1 x size x size的的矩阵，该方法在训练中构建tgt_mask时，即遮盖住翻译的正确答案时使用。
    # 前面加1是为了和tgt的tensor维度保持一致，因为tgt第一维时batch_size
    attn_shape = (1, size, size) 
    # 
    """
    subsequent_mask：首先通过triu函数生成上三角阵，当size为5时，结果为：
       [[[0., 1., 1., 1., 1.],
         [0., 0., 1., 1., 1.],
         [0., 0., 0., 1., 1.],
         [0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 0.]]]
    """
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0 # 然后将0变为1，将1变为0，其中0为需要mask掉的内容

# 拆成多头之后的计算
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    
    """
    Multi-Head Attention：
    shape: [batch, head数, 词数, d_model/head数]，所谓多头就是将d_model拆开
    """
    d_k = query.size(-1) # 获取d_model的值，query与输入的shape相同
    """
    q: [batch, head, 词数, d_model/head数], k: [batch, head, d_model/head数, 词数]
    q*k: [batch, head, 词数, 词数]，然后/math.sqrt(d_k)，其维度不变
    则scores为方阵，其shape: [batch_size, head数, 词数, 词数]
    
    src_mask和tgt_mask比较：
    src_mask的shape为(batch_size, 1, 1, 词数)，tgt_mask的shape为(batch_size, 1, 词数, 词数)
    src_mask在最后一维对非句子的词mask，tgt_mask需要方阵一次性对所有的训练句子mask
    """
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
    if mask is not None:
        scores = scores.masked_fill(mask.cuda() == 0, -1e9)   # pad值为-1e9，softmax之后为0
    p_attn = scores.softmax(dim=-1) # p_atten: shape: [batch, head数, 词数, 词数]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # attention公式结果: [batch, head数, 词数，d_model/head数]，p_attn方便将注意力分数可视化

 # 将Q,K,V最后一个纬度拆分，拆成多个head，代入attention，最后将多头融合
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 定义W^q, W^k, W^v和W^o矩阵(用于合并)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None): 
        # query, key, value将与W^q, W^k, W^v矩阵相乘
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            # mask与query维度一致，shape:[batch, head数, 词数, d_model/head数]
            mask = mask.unsqueeze(1) 
        nbatches = query.size(0)    # 获取batch_size

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            # Q,K,V:[batch, head数, 词数,d_model/head数]
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # x:[batch, head数, 词数, d_model/head数],self.attn:[batch, head数, 词数, 词数]
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        # x:[batch, head数, 词数,d_model/head数] -> x:[batch, 词数, d_model]
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        # 最终通过W^o矩阵再执行一次线性变换，得到最终结果
        return self.linears[-1](x).cuda()

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1): # d_ff为隐层神经元数量
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu())).cuda() 

# 两个embedding层使用相同的权重    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x.cuda()) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x为embedding后的inputs,例如(1,7,512),batch size为1,7个单词,单词维度为512
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

    
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy   # 将模型深拷贝一份，相当于new一个全新的模型
    # 1、构建多头注意力机制
    attn = MultiHeadedAttention(h, d_model).cuda()
    # 2、构建feed forward网络
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).cuda()
    position = PositionalEncoding(d_model, dropout).cuda()
    # 3、构建transformer模型
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N).cuda(),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).cuda(), N).cuda(),
        nn.Sequential(Embeddings(d_model, src_vocab).cuda(), c(position).cuda()),
        nn.Sequential(Embeddings(d_model, tgt_vocab).cuda(), c(position).cuda()),
        Generator(d_model, tgt_vocab).cuda(), # 用于预测下一个token
        d_model
    ).cuda()

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    # 模型参数初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

"""
# 推理(inference) 在训练之前用来测试模型，没有给出损失函数，因此输出是随机的
    # 原词典和目标词典大小都为11，encoderlayer和decoderlayer都是2层
    test_model = make_model(11, 11, 2)
    test_model.eval()
    # 定义inputs，shape为[1, 10]，即一个句子，每个句子10个单词
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    # 定义src_mask，即所有的单词为1，都是有效的，没有pad
    src_mask = torch.ones(1, 1, 10)

    # inputs经ecoder之后，得到memory
    memory = test_model.encode(src, src_mask)
    # 初始化真实值为[[0]]，用于保存预测结果，其中0表示'bos'
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        # ys.size(1)指第二个维度有几个数据，有几个单词就生成(1, size, size) 的下三角为1的mask矩阵。.size()函数如下所示：
        # a = torch.tensor([[1,2,3], [4,5,6]])
        # print(a.size(0))    # 2 第0维有[1,2,3]和[4,5,6]两个数据
        # print(a.size(1))    # 3 第1维有1，2，3（或4，5，6）三个数据
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data) 
        )
        # out维度变化的是[batch_size, 词数，词向量]中词数这个维度（decoder一个一个输出词），因此每次只取最后一个词
        prob = test_model.generator(out[:, -1])
        # _即最大值，next_word即最大值对应的索引
        _, next_word = torch.max(prob, dim=1)
        # 取出预测结果
        next_word = next_word.data[0]
        # 将本次的预测结果与之前的预测结果进行拼接，作为之后decoder的输入
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)
"""

# 定义一个batch，存放一个batch的src，tgt，src_mask等对象，方便后续使用
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src  # 与EncoderDecoder中forward函数的src一致
        self.src_mask = (src != pad).unsqueeze(-2) # 将src中pad部分掩盖，使维度与scores保持一致
        if tgt is not None: # tgt即目标句子
            self.tgt = tgt[:, :-1] # 需要去掉最后一个词，tgt存储的是decoder的输入，所以不会出现最后一个词，即`<bos> 我 爱 你`（没有'<eos>'）
            self.tgt_y = tgt[:, 1:] # tgt_y存储希望预测的结果，即decoder的预测输出结果，去掉第一个词'<bos>'，即“我 爱 你 <eos>”
            self.tgt_mask = self.make_std_mask(self.tgt, pad) # tgt_mask & subsequent_mask
            self.ntokens = (self.tgt_y != pad).data.sum() # 该batch的tgt_y的总token的数量，去除'pad'部分的词数，同时也去掉了前面的'<bos>'

    @staticmethod
    def make_std_mask(tgt, pad): # 生成tgt_mask
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2) # 生成非句子成分的mask，加入pad
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as( 
            tgt_mask.data
        )
        return tgt_mask
    
# 用于保存一些训练状态
class TrainState:  
    """Track number of steps, examples, and tokens processed"""
    # 一个batch算一次，或者一次loss.backward()算一次。可能累计多次loss然后进行一次optimizer.step()
    step: int = 0  # Steps in the current epoch
    # 模型参数更新的次数，即optimizer.step()的次数
    accum_step: int = 0  # Number of gradient accumulation steps
    # 记录训练过的样本数量
    samples: int = 0  # total # of examples used
    # 记录处理过的target的token数量
    tokens: int = 0  # total # of tokens processed
    
# 进行一个epoch训练
def run_epoch(
    data_iter,  # 可迭代对象，一次返回一个batch对象
    model,      # transformer模型，EncoderDecoder类对象
    loss_compute,   # SimpleLossCompute对象，用于计算损失
    optimizer,  # Adam优化器，验证时的optimizer是DummyOptimizer
    scheduler,  # LambdaLR对象，用于调整Adam的学习率，实现WarmUp。验证时的scheduler是DummyScheduler
    mode="train",
    accum_iter=1, # 多少个batch更新一次参数，默认为1，每个batch都对参数进行更新
    train_state=TrainState(), # TrainState对象，用于保存一些训练状态
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0    # 记录tgt_y(去掉bos)的总token数，用于对total_loss进行正则化
    total_loss = 0      
    tokens = 0          # 记录target的总token数，每次打印日志后清零
    n_accum = 0         # 本次epoch的参数更新次数
    for i, batch in enumerate(data_iter):
        out = model.forward(    # out是decoder输出，generator的调用放在了loss_compute中
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask 
        ).cuda()
        """
        计算损失，传入的三个参数分别为：
        1. out: EncoderDecoder的输出，该值并没有过最后的线性层，该线性层被集成在了计算损失中
        2. tgt_y: 要被预测的所有token，例如src为`<bos> I love you <eos>`，则`tgt_y`则为
                  `我 爱 你 <eos>`
        3. ntokens：这批batch中有效token的数量。用于对loss进行正则化。

        返回两个loss，其中loss_node是正则化之后的，所以梯度下降时用这个。而loss是未进行正则化的，用于统计total_loss。
        """
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward() # 计算梯度
            
            train_state.step += 1 # 计算step次数
            train_state.samples += batch.src.shape[0] # 记录样本数量，batch.src.shape[0]获取的是batch_size
            train_state.tokens += batch.ntokens # 记录处理过的token数
            
            if i % accum_iter == 0: # 如果达到了accum_iter次，就进行一次参数更新
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                n_accum += 1    # 记录本次epoch的参数更新次数
                train_state.accum_step += 1 # 记录模型的参数更新次数Epoc
            scheduler.step()    # 更新学习率

        total_loss += loss  # 累计loss
        total_tokens += batch.ntokens   # 累计处理过的tokens
        tokens += batch.ntokens # 累计从上次打印日志开始处理过的tokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"): # 每40个batch打印一次日志
            lr = optimizer.param_groups[0]["lr"]    # 打印当前的学习率
            elapsed = time.time() - start   # 记录40个batch消耗的时间
            """        打印日志
            i: 本次epoch的第几个batch
            n_accum: 本次epoch更新了多少次模型参数
            loss / batch.ntokens: 对loss进行正则化，然后再打印loss，其实这里可以直接用loss_node
            tokens / elapsed: 每秒可以处理的token数
            lr: 学习率（learning rate），这里打印学习率的目的是看一下warmup下学习率的变化。
            注意：学习率过低，损失函数的变化速度就越慢，容易过拟合。学习率过高，容易发生梯度爆炸，loss振动幅度越大，模型难以收敛。
            """
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time() # 重置开始时间
            tokens = 0  # 重置tokens数
        del loss
        del loss_node
    return total_loss / total_tokens.cuda(), train_state   # 返回正则化之后的total_loss，返回训练状态

# 学习率调整函数
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:   # 避免分母为0
        step = 1
    # 学习率的计算公式，具体见https://blog.csdn.net/zhaohongfei_358/article/details/126085557
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
    
# 标签平滑（正则化，减少过拟合） + 计算损失
# 在训练时即假设标签可能存在错误，避免“过分”相信训练样本的标签。之后计算loss的时候，使用平滑后的标签。当目标函数为交叉熵时，称为标签平滑（Label Smoothing）。平滑指的是把两个极端值0和1变成两个不那么极端的值，如[0,0,0,1,0]平滑后的label为[0.05,0.05,0.05,0.8,0.05]
class LabelSmoothing(nn.Module):
    """
    size: 目标词典的大小。
    padding_idx: 空格('<blank>')在词典中对应的index，`<blank>`等价于`<pad>`，LabelSmoothing的时候不应该让<pad>参与。
    smoothing: 平滑因子，0表示不做平滑处理
    """
    def __init__(self, size, padding_idx, smoothing=0.0): 
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum") # 定义损失函数KL散度，是一种多分类常用的损失函数，官方文档见https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None # true distribution，平滑后的标签

    def forward(self, x, target):
        """
        x: generator输出的概率分布。Shape为(batch_size, 词典大小)
        target: 目标标签。Shape为(batch_size)
        """
        assert x.size(1) == self.size   # 确保generator的输出维度与词典大小一致，否则计算loss会出错
        
        true_dist = x.data.clone().cuda()   # 创建一个与x有相同shape的tensor
        true_dist.fill_(self.smoothing / (self.size - 2)) # 将true_dist全部填充为 self.smoothing / (self.size - 2)
        true_dist.scatter_(1, target.data.unsqueeze(1).cuda() , self.confidence)  # 将true_dist的dim=1上与target.data.unsqueeze(1)索引对应的值变为src
        true_dist[:, self.padding_idx] = 0 # 将空格所在的index填充为0
        mask = torch.nonzero(target.data == self.padding_idx) # 找出target中label为<blank>的标签
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze().cuda() , 0.0) # 将"<blank>"所在的label整个设置为0
        self.true_dist = true_dist # 保存平滑后的标签
        # 使用平滑后的标签计算损失，由于对`<blank>`部分进行了mask，所以在这部分不参与损失计算
        return self.criterion(x, true_dist.clone().detach()) 

# 造数据    
def data_gen(V, batch_size, nbatches): 
    """
    生成一组随机数据
    V: 词典的大小
    batch_size：有多少句话
    nbatches: 一共输入nbatch个batch完成一轮epoch
    return: yield一个Batch对象
    """
    for i in range(nbatches): # 遍历batch，生成nbatch个batch
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1 # 将每行的第一个词改为1，即"<bos>"
        src = data.requires_grad_(False).clone().detach() # 该数据不需要梯度下降
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0) # yield返回值后，继续执行函数体内代码，返回一个Batch对象

# 损失计算 + generator部分的前向传递        
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator # 根据decoder的输出预测下一个token
        self.criterion = criterion  # labelsmoothing对象，对label进行平滑和计算损失

    def __call__(self, x, y, norm):
        """
        x: Decoder的输出，还没有进入generator，x:[batch_size, 词数, d_model]
        y: batch.tgt_y，要被预测的所有token，例如src为`<bos> I love you <eos>`，则`tgt_y`则为`我 爱 你 <eos>`，即去掉<bos>的目标句子
        norm: batch.ntokens, tgt_y中的有效token数。用于对loss进行正则化。
        """
        # x为调用generator输出的概率分布(EncoderDecoder的forward中并没有调用generator)
        x = self.generator(x) 
        # 使用KL散度计算损失，然后/norm对loss进行正则化，防止loss过大过小，取其平均数。
        sloss = (
            self.criterion(
                # x:[batch_size, 目标词典大小]    y:[batch_size]
                # generator一次性得到所有词在词典中的概率分布，y就是这些词的目标翻译句子
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        ).cuda()
        return sloss.data * norm, sloss

# 翻译任务的预测：先求出encoder的输出memory，然后利用memory一个一个求出token，与前面的inference_test()代码相同
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    进行模型推理，推理出所有预测结果
    model: Transformer模型，即EncoderDecoder类对象
    src: Encoder的输入inputs，Shape为(batch_size, 词数)。例如：[[1, 2, 3, 4, 5, 6, 7, 8, 0, 0]]，即一个句子，该句子有10个词，分别为1,2,...,0
    src_mask: src的掩码，掩盖住非句子成分，将其填充到等长
    max_len: 一个句子的最大长度
    start_symbol: '<bos>' 对应的index，在本例中始终为0
    return: 预测结果，例如[[1, 2, 3, 4, 5, 6, 7, 8]]
    """
    memory = model.encode(src, src_mask).cuda() # memory为encode最后一层的输出
    # 初始化ys为[[0]]，用于保存预测结果，其中0表示'<bos>'
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data).cuda()
    # 循环调用decoder，一个个进行预测，直到decoder输出"<eos>"或达到句子最大长度
    for i in range(max_len - 1):
        # 将memory和decoder之前的输出作为参数，预测下一个token
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        ).cuda()
        # 每次取decoder最后一个词的输出送给generator进行预测
        prob = model.generator(out[:, -1]) 
        # 取出数值最大的那个，它的index在词典中对应的词就是预测结果
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]   # 取出预测结果
        # 将这一次的预测结果和之前的拼到一起，作为之后decoder的输入
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word).cuda()], dim=1
        )
    return ys   # 返回最终的预测结果

# 训练一个简单的copy任务
def example_simple_model():
    # 定义词典大小
    # V = 11  
 
    # 定义损失函数
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # 构建模型，src和tgt的词典大小都为V，N为encode和decode层数
    model = make_model(V, V, N=2).cuda()

    # 使用Adam优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    
    #自定义Warmup学习率
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400 # wanrmup400次，即从第400次学习率开始下降
        ),
    )

    # 传入的参数是指向文件夹的路径，之后需要使用writer对象取出的任何数据都保存在这个路径之下
    # image_path = r"D:\Master_Files\SC\Experiment\image\result1"
    # writer = SummaryWriter(image_path)
    
    for epoch in range(100): # 运行几个epoch
        model.train() # 将模型调整为训练模式
        loss1 = SimpleLossCompute(model.generator, criterion)
        loss2 = SimpleLossCompute(model.generator, criterion)
        run_epoch(  # 训练一个Batch
            data_gen(V, batch_size, 20), # 生成20个batch对象进行训练
            model.cuda(),
            # SimpleLossCompute(model.generator, criterion),
            loss1,
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval() # 进行一个epoch后进行模型验证
        run_epoch(
            data_gen(V, batch_size, 5), # 生成5个对象进行验证
            model.cuda(),
            # SimpleLossCompute(model.generator, criterion),
            loss2,
            DummyOptimizer(), # 验证时不进行参数更新
            DummyScheduler(), # 验证时不调整学习率
            mode="eval",
        )[0].cuda() # run_epoch返回loss和train_state，这里只取loss，所以是[0]。但是源码中没有接收这个loss，所以[0]没有实际意义


    # 生成x轴跨度为100的折线图，y轴坐标代表每一个epoch的损失函数，这个折线图会保存在指定的路径下
    # writer.add_scalar('loss1', loss1.avg, epoch)
    # writer.add_scalar('loss2', loss2.avg, epoch)
    
    # 将模型调整为测试模式，准备开始copy任务
    model.eval()
    # 定义src为0-9，看看模型能否重新输出0-9
    src = torch.LongTensor([[0, 2, 2, 3, 4, 5, 6, 7, 8, 9]])
    # 句子的最大长度是src第二维的值
    max_len = src.shape[1]
    # 不需要mask，因为这10个都是有意义的数字
    src_mask = torch.ones(1, 1, max_len)
    # 使用greedy_decode进行推理
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0)) # Loss:0.11     tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # 获取下标对应的值
    IDIndex = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0)
    IDIndex = IDIndex.detach().cpu().numpy()  # 将tensor变量转换为数组
    getData(IDIndex)


execute_example(example_simple_model)