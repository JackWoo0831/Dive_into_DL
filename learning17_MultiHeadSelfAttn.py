'''
多头自注意力 无需多说
在之前注意力模型的基础上,query key和value都是输入X,这称作"自"注意力
位置编码有很多种方式,如果采用三角函数进行编码,好处是利用三角函数的周期性,可以学习相对位置,
而忽略绝对位置.
'''
# In[]
import torch 
from torch import nn 

from learning16_Attention import DotProductAttn


# 将qkv的shape改变,来分成多个head
def transpose_qkv(X,num_heads):
    '''
    input:X:q or k or v,shape:(batch_size,num,num_hiddens)
    heads:number of heads
    output:X',shape:(batch_size * num_heads,num,num_hiddens/num_heads)
    '''
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1) # (bs,num,num_heads,num_hiddens/num_heads)

    X = X.permute(0,2,1,3) # (bs,num_heads,num,num_hiddens/num_heads)

    return X.reshape(-1,X.shape[2],X.shape[3]) # (batch_size * num_heads,num,num_hiddens/num_heads)

# 对transpose_qkv的逆操作
def transpose_output(X, num_heads):

    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3) # (bs,num,num_heads,num_hiddens/heads)
    return X.reshape(X.shape[0], X.shape[1], -1) # (bs,num,num_hiddens)



# 定义多头注意力类
class MultiHeadAttn(nn.Module):
    def __init__(self,key_size, query_size,value_size,num_hiddens,num_heads,dropout,bias = False):
        super().__init__()

        self.num_heads = num_heads # heads number
        self.attention = DotProductAttn(dropout) # 用的是点积注意力

        # 定义qkv矩阵
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)

        # 计算qkv后的output matrix
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self,queries, keys, values, valid_lens = None):
        # 其实基本思路是这样的:
        # 多头注意力,本应该分头计算,然后concat起来.但是可以先把头的数量乘到bs上,为了用之前定义的DPAttn
        # 就应把qkv都整成3D tensor(bs,num,num_hiddens) 进行DPAttn
        # 这样计算完之后,再把head分离出来 恢复到原来的shape 与W_o乘

        # 将qkv都转为(batch_size * num_heads,num,num_hiddens/num_heads)

        
        queries = transpose_qkv(self.W_q(queries),self.num_heads)
        keys = transpose_qkv(self.W_k(keys),self.num_heads)
        values = transpose_qkv(self.W_v(values),self.num_heads)

        if valid_lens is not None:
            # valid_lens shape:(bs,)
            # 在轴0，将第一项（标量或者矢量）复制`num_heads`次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # DPAttn 
        output = self.attention(queries,keys,values,valid_lens)

        # 将shape恢复
        output_concated = transpose_output(output,self.num_heads)
        
        # print(type(self.W_o(output_concated)))
        return self.W_o(output_concated)

        

   



#In[]

#test
num_hiddens, num_heads = 100, 5



attention = MultiHeadAttn(num_hiddens,num_hiddens,num_hiddens,num_hiddens,num_heads,0.2)

print(attention.eval())

batch_size, num_queries = 32, 8

valid_lens = torch.rand(size=(batch_size,))

X = torch.ones((batch_size, num_queries, num_hiddens))

attention(X,X,X,valid_lens).shape
# %%
