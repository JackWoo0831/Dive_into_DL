'''
注意力机制
最初源于心理学的概念 query相当于随意注意,key相当于已经有的东西(非随意注意),value是
query和key经过作用后,经过注意力池化,得到的
最基本的注意力范式是:
f(x) = \sum_{i} \alpha(x,x_i) * y_i
其中x是query,x_i是key,y_i是value,\alpha是函数,其实就是注意力权重

'''

#In[]

from Common_functions import mask_softmax

import torch 
from torch import nn 
import math


'''
1.加性注意力 AdditiveAttention
当query和key的长度不同时,可以使用加性注意力,公式:
\alpha(q,k) = w_v^T * tanh(W_q*q + W_k * k)
其中q:(q,) k:(k,) W_q:(h,q) W_k:(h,k) w_v:(h,)

相当于把q和k连接(concat)起来输入一个MLP,MLP包含一个隐藏层,单元数为h
'''
class AdditiveAttention(nn.Module):
    def __init__(self,key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__()

        #别忘了禁用bias
        self.W_k = nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=False)
        self.w_v = nn.Linear(num_hiddens,1,bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,queries, keys, values, valid_lens = None):
        '''
        queries:(batch_size,num_q,q,)
        keys:(batch_size,num_k,k,)
        values:(batch_size,num_k,v,) value的数量和key相同,不相同最后一步乘法出问题
        '''
        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries:(batch_size,num_q,h,)
        # keys:(batch_size,num_k,h,)
        # 对每个query,考虑所有的key,因此要采用广播机制
        # queries.unsqueeze(2):(batch_size,num_q,1,h,)
        # keys.unsqueeze(1):(batch_size,1,num_k,h,)
        feature = queries.unsqueeze(2) + keys.unsqueeze(1)
        # feature:(batch_size,num_q,num_k,h)
        feature = torch.tanh(feature)
        scores = self.w_v(feature)
        # scores:(batch_size,num_q,num_k,1)
        # 因为是1,去掉最后一个维度
        scores = scores.squeeze(-1)

        # 结果再经过softmax 
        self.attention_weights = mask_softmax(scores,valid_len=valid_lens)
        # attention_weights:(batch_size,num_q,num_k)

        # 将注意力分数和values相乘
        return torch.bmm(self.dropout(self.attention_weights), values)






#In[]
# test  
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1) #values:(2,10,4)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)

#In[]

'''Scaled Dot product attention
点积效率更高,但前提是q和k的维数相同.公式:
score = softmax(\frac{QK^T}{\sqrt{d}})V
其中:Q:(num_q,q) K:(num_k,k) V:(num_k,v) d=q
score:(num_q,v)
'''
class SDotProductAttn(nn.Module):
    def __init__(self,dropout,**kargs):
        super().__init__()
        # 需要学的只有Dropout 不要和self attention混淆了
        self.dropout = nn.Dropout(dropout)

    def forward(self,queries,keys,values,valid_lens = None):
        '''
        输入shape和加性注意相同
        '''
        d = queries.shape[-1]

        score = torch.bmm(queries,keys.transpose(1,2)) / math.sqrt(d)

        self.attention_weights = mask_softmax(score, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

