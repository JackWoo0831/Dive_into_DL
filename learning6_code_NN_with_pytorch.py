#In[]
'''
本节笔记:

'''

import torch 
from torch import nn 
from torch.nn import functional as F


#通过继承nn.Module类来自定义一个block,继承父类可以继承许多方法,注意必须写super().__init__()
class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim,256)
        self.output = nn.Linear(256,output_dim)

    def forward(self,X):
        
        return self.output(F.relu(self.hidden(X)))


#test一下
input_dim, output_dim = 20, 10
X = torch.rand(2,input_dim) 

net = MLP(input_dim,output_dim)

print(net(X))
#In[]

#之前我们用如下方式来定义一个网络

net_API = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_dim,256),
    nn.ReLU(),
    nn.Linear(256,output_dim))


#自己实现一个Sequential

class Mysequential(nn.Module):

    def __init__(self,*args):
        super().__init__()
        for block in args:
            self._modules[block] = block
        
        #print(type(self._modules),type(self._modules.values()))#<class 'collections.OrderedDict'> <class 'odict_values'>

    def forward(self,X):
        for block in self._modules.values():
            X = block(X)

        return X


net = Mysequential(
    nn.Flatten(),
    nn.Linear(input_dim,256),
    nn.ReLU(),
    nn.Linear(256,output_dim)
)

net(X)
#In[]
'''
重要:参数管理,主要包括:
1.访问参数
2.参数初始化
3.共享参数
'''

#考虑之前的net_API网络 Flatten-Linear-ReLU-Linear
print(net_API[3].state_dict())
#In[]

print(type(net_API[3].bias)) #<class 'torch.nn.parameter.Parameter'>

print(net_API[3].bias.data) #访问data

print(net_API[3].weight.grad == None) #除了访问data外,还能访问梯度


#In[]

#访问所有参数,采用named_parameters()方法

print(*[(name, param.shape) for name, param in net_API[3].named_parameters()])
print(*[(name, param.shape) for name, param in net_API.named_parameters()]) #*表示递归访问,Flatten和ReLU没有参数


#In[]
#也可以用state_dict()+name的方式来访问值

print(net_API.state_dict()['3.bias'].data)



#In[]

#block之间的嵌套

def block1():
    return nn.Sequential(nn.Linear(20, 128), nn.ReLU(),
                         nn.Linear(128, 20), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1()) #嵌套了四遍block1
    return net

rgnet = nn.Sequential(block2(), nn.Linear(20, 1))
rgnet(X)


print(rgnet)

#可以观察rgnet.state_dict()发现,例如要访问block中第2个block1的第二个线性层的bias,应该是:
print(rgnet.state_dict()['0.block 1.2.bias'].data)
#In[]
'''
重要
参数绑定(共享)
'''

shared = nn.Linear(8,8) #共享的层

net = nn.Sequential(
    nn.Linear(20,8),
    nn.ReLU(),

    shared,
    nn.ReLU(),

    shared,
    nn.ReLU(),

    nn.Linear(8,2)
)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象,而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])



#In[]
'''
自定义layer
'''

#自定义一个layer,作用是中心化
class CenteredLayer(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self,X):

        return X - X.mean()


layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
#In[]

#自定义一个Linear Layer
class MyLinearLayer(nn.Module):

    def __init__(self,input_dim,output_dim):
        super().__init__()

        #进行参数初始化

        self.weight = nn.Parameter(torch.randn(size=(input_dim,output_dim)))
        self.bias = nn.Parameter(torch.zeros(size=(output_dim,))) #逗号是为了使之成为列向量


    def forward(self,X):

        return torch.matmul(X,self.weight.data) + self.bias.data #matmul张量乘法,mm只能是二维矩阵乘法

myl = MyLinearLayer(5,3)

myl.weight.data

#forward计算
myl(torch.rand(size=(2,5)))


# %%

