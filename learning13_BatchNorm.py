'''
Batch Norm解决的问题:梯度消失.
由于梯度一般都比较小,当网络很深的时候,反向传播时模型的前面部分的梯度就会很小,从而参数更新会变慢,因此train不动.
模型后面train的动,前面train不动就会造成一种不平衡.当输入变化较大的时候,后面就会收敛的很慢.
一般应用在卷积层之后,激活函数之前或者卷积层之前

Batch Norm的做法:
对某层卷积层的通道维或者FC层的特征维计算均值与方差:
\mu = 1/|B| * \sum_{i \in B}x_i 
\sigma^2 = 1/|B| * \sum_{i \in B}(x_i - \mu)^2 + e
其中B为该层输出的序列集合,e为小正数,防止方差为0

之后按下式更新输出x_i:
x_i = \gamma * \frac{x_i - \mu}{\sigma} +\beta
其中gamma和beta是可学习的参数
可见,BatchNorm是一个线性层

BatchNorm通过加入扰动(每个每次batch的均值和方差是随机的)来降低模型复杂度,进而提升效果,因此没必要和Dropout一起用
'''
#In[]
import torch 
from torch import nn 
from torch.nn import functional as F

from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import resize

import Common_functions


#In[]

#定义batch_nrom函数
def batch_nrom(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    '''
    input:X:input tensor
    gamma,beta:param to learn
    moving_mean,moving var:global mean and var for inference,respectively
    eps:small positive
    momentum:use to update gamma and beta,sliding update

    output:X_:result 
    moving_mean,moving_var: updated moving mean and var for next forward
    '''

    #判断是否在inference 没有梯度就是在做inference
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps) 

    else:#如果在训练
        assert len(X.shape) in (2,4) #断言X的shape长度要么是2(Linear)要么是4(Conv)
        if len(X.shape) == 2: #如果是线性层
            mean = X.mean(dim=0) #线性层的维度是(batchsize,feature).dim=0是对特征求平均
            var = ((X - mean) ** 2).mean(dim = 0) 

        else:
            #2D conv的情形计算通道维度上的均值和方差
            mean = X.mean(dim = (0,2,3),keepdim = True) #通道维是第1维 保持X的shape便于广播机制
            var = ((X - mean) ** 2).mean(dim = (0,2,3),keepdim = True) 

        #计算出mean和var后,做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)

        #更新moving_mean和moving_var 它俩利用滑动平均值来更新beta和gamma 其实就是不仅考虑当下的均值方差,还考虑之前的均值方差
        #(动量的思想)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean
        moving_var = momentum * moving_var + (1 - momentum) * var

    #输出更新的X

    X_ = gamma * X_hat + beta

    return X_,moving_mean,moving_var




#定义BatchNorm层

class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super().__init__()
        #初始化参数
        #要注意gamma等参数的shape,为了应用广播机制
        if num_dims == 2:#Linear
            shape = (1,num_features)
        else:#conv
            shape = (1,num_features,1,1)

        self.gamma = nn.Parameter(torch.ones(size=shape)) #可学习的参数,定义为nn.Parameter
        self.beta = nn.Parameter(torch.zeros(size=shape))

        self.moving_mean = torch.zeros(size=shape)
        self.moving_var = torch.ones(size=shape)

    def forward(self,X):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y,self.moving_mean,self.moving_var = batch_nrom(
            X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)

        return Y


#In[]

#将BatchNorm应用在Lenet
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

#In[]
#在FashionMNIST上train

#加载数据
trans = transforms.ToTensor() #class,将PIL图片或numpy.ndarray格式转为tensor的类

mnist_train = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=True,transform=trans,download=False)
mnist_test = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=False,transform=trans,download=False)

batch_size = 256

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)


#训练
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

Common_functions.train_device(net,train_iter,test_iter,lr=1.0,device=device)
# %%
