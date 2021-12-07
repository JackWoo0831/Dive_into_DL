#In[]
'''
本节笔记:
对FC层加以限制:平移不变性和局部性就成为了卷积层.
卷积层的本质就是权值共享的全连接层
'''
import torch 
from torch import nn 

#定义二维卷积运算
def conv2d(X,Kernel):
    '''
    input:
    X:input,torch.Tensor
    Kernel:torch.Tensor
    output:
    y
    no padding and stride
    '''

    h, w = Kernel.shape

    y = torch.zeros(size=(X.shape[0] - h + 1,X.shape[1] - w + 1)) #初始化结果,注意结果的shape,可以举例子来计算

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):

            y[i,j] = (X[i:i + h,j:j + w] * Kernel).sum()

    return y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
conv2d(X, K)

#In[]

#自定义卷积层

class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(size=kernel_size))

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,X):

        return conv2d(X,self.weights.data) + self.bias




#In[]

#学习(训练)卷积核

#定一个X
X = torch.ones((6, 8))
X[:, 2:6] = 0

#我们假定一个Kernel的GT值:
K = torch.tensor([[1.0, -1.0]])

#GT output:
Y = conv2d(X,K)


#pytorch API
conv2d_API = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,2),bias=False)

#将输入输出reshape成(N,Channnels,H,W)

X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))

for i in range(30):
    Y_hat = conv2d_API(X)

    l = (Y_hat - Y) ** 2

    conv2d_API.zero_grad() #梯度清零

    l.sum().backward()

    conv2d_API.weight.data[:] -= 0.03  * conv2d_API.weight.grad #SGD

    print(f'bacth:{i + 1},loss:{l.sum()}')


print(conv2d_API.weight.data)

#In[]
'''
Stirde and padding
'''
import torch 
from torch import nn 

#没啥可说的 看整理的博客


#In[]
'''
multi channels in input and output
'''

#多输入通道:例如RGB图片的输入维数是3维,相应地有3个kernel(也可认为kernel是3dTensor),是分别在每个维度卷积后再求和

def Conv2d_multi_input(X,kernel):
    '''
    input:
    X:c_in x H x W
    kernel:c_in x k_0 x k_1
    '''

    return sum(conv2d(x,k) for x,k in zip(X,kernel)) #zip在第一个维度整合


#测试
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

Conv2d_multi_input(X,K)


#In[]

#多输出通道:其实就是用不同数量的卷积核来识别不同的模式
#e.g.:
#input:X:c_in x H x W kernel: c_out x c_in x k_0 x k_1
#output:c_out x H' x W'

def Conv2d_multi_in_out(X,kernel):
    '''
    input:X:c_in x H x W
          kernel:c_out x c_in x k_0 x k_1
    output:
    c_out x H' x W'
    '''

    return torch.stack([Conv2d_multi_input(X,k) for k in kernel],0) #for k in kernel是按第一个维度遍历kernel,stack在新维度上进行拼接(concatenate)

#将kernel改为4Dtensor
K = torch.stack((K, K + 1, K + 2), 0)


Conv2d_multi_in_out(X, K)
#In[]

'''
Pooling
池化解决卷积对位置敏感的问题
'''

import torch 
from torch import nn
from math import floor 

def my_pooling2d(X,pool_size,stride=None,padding=0,mode='max'):
    '''
    input: X:c_in x H x W
    pool_size:k_0 x k_1
    mode:'max' or 'avg'
    note:ignore dilation and padding mode

    output:c_in x H' x W' (pooling does not change the channel domin)
    
    '''
    #池化和卷积操作的本质相同,只是池化不改变channel维度,通过练习stride和padding,加深理解

    #先处理padding 假设H和W的padding相同 假设padding的模式是填充0
    C,H,W = X.shape
    k_0,k_1 = pool_size

    X_ = torch.zeros(size=(C,H + 2*padding,W + 2*padding))
    X_[:,padding:X_.shape[1]-padding,padding:X_.shape[2]-padding] = X #相当于在X周围填充0


    #开始pooling

    #默认stride的大小跟pool_size相同,这样不重叠
    if stride == None:
        stride = pool_size[0]

    #输出Y:
    H_,W_ = floor((H + 2*padding - k_0)/stride + 1),floor((W + 2*padding - k_1)/stride + 1)
    Y = torch.zeros(size=(C,H_,W_))


    h = 0

    while h < H_:
        w = 0
        while w < W_:
            
            if mode == 'max':
                for c in range(C):
                    Y[c,h,w] = X_[c,h:h + k_0,w:w + k_1].max()
            elif mode == 'avg':
                for c in range(C):
                    Y[c,h,w] = X_[c,h:h + k_0,w:w + k_1].mean()
            else:
                pass

            w += stride
        h += stride


    return Y


#test
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]])

Y = my_pooling2d(X,pool_size=(2, 2),stride=1)

print(Y)


#使用API实现:

#nn.MaxPool2d(kernel_size=(3,3),stride=3,padding=1,dilation=2)     



    


    





    
# %%
