#In[]
import torch
from torch.serialization import load
#In[]
'''自动求导演示'''
#创建一个tensor
x = torch.arange(4.0)
x

x.requires_grad_(True) #存储梯度
#In[]
y = 2 * torch.dot(x,x)

y.backward() #y对x求导
x.grad #查看导数 应改为4x

print(x.grad == 4 * x) #验证导数值

#In[]
#计算x.sum()的梯度
#pytorch会默认累加梯度,因此需要先清零
x.grad.zero_()

#定义y
y = x.sum()

#计算导数
y.backward()
x.grad

#In[]
#将某些计算移动到记录的计算图之外
x.grad.zero_()
y = x * x #y此时是一向量,x的对应元素相乘
u = y.detach() #detach意为解除y跟x的关系,将y认为是一个新的向量
z = u * x

z.sum().backward() #此时u跟x无关,导数应为u
print(x.grad == u)


#In[]
'''线性回归'''
import random
import torch
from d2l import torch as d2l

#定义函数生成人造数据集
def synthetic_data(w,b,num_examples):
    #生成y = wX + b + noise
    X = torch.normal(0,1,(num_examples,len(w))) #生成均值为0,标准差为1的正态分布,shape为(样本数量,w的维数)

    #必须转为float,否则报错
    X = X.float()
    w = w.float()
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape) #加入噪声
    return X,y.reshape((-1,1))  

#规定w和b的真值
true_w = torch.tensor([2,-3,4])
true_b = 4.2

features,lables = synthetic_data(true_w,true_b,1000)
#In[]
#定义一个data_iter函数,其将数据分成小batch

def data_iter(batch_size,features,labels):
    num_examples = len(features) #样本数量
    indices = list(range(num_examples)) #产生索引

    random.shuffle(indices) #打乱标签 随机读取
    for i in range(0,num_examples,batch_size): #从0开始 以batch_size为步长 分割数据
        batch_indices = torch.tensor(indices[i:min(i + batch_size - 1,num_examples)]) #建立本batch的索引
        yield features[batch_indices],labels[batch_indices] #yield让函数继续执行且返回 此时函数被视为一个迭代器

#随机初始化参数
w = torch.normal(0,0.01,size=(3,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

#定义模型
def linreg(X,w,b):
    return torch.matmul(X,w) + b

#定义损失函数
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2 #reshape是防止出现行列向量不匹配的情况

#定义优化算法 随机梯度下降
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size #SGD:w_t = w_{t-1} - lr * \partial l / \partial w 
            #此处/batch_size是因为损失函数那里没有除
            param.grad.zero_()

#In[]
#训练过程

#定义训练参数
lr = 0.03
epochs = 3
batch_size = 10

for epoch in range(epochs):
    for X,y in data_iter(batch_size,features,lables):
        l = squared_loss(linreg(X,w,b),y)#计算loss
        #print(l.shape)

        l.sum().backward() #计算l关于w,b的梯度
        sgd([w,b],lr,batch_size)
    
    with torch.no_grad():
        train_l = squared_loss(linreg(features,w,b),lables) #计算本epoch更新之后的损失
        print(f'epoch{epoch + 1},loss{float(train_l.mean()):f}')

#In[]
'''使用torch实现线性回归'''
import numpy as np 
import torch 
from torch.utils import data
import torch.nn as nn
from d2l import torch as d2l

#定义函数生成人造数据集
def synthetic_data(w,b,num_examples):
    #生成y = wX + b + noise
    X = torch.normal(0,1,(num_examples,len(w))) #生成均值为0,标准差为1的正态分布,shape为(样本数量,w的维数)

    #必须转为float,否则报错
    X = X.float()
    w = w.float()
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape) #加入噪声
    return X,y.reshape((-1,1))  


#规定w和b的真值
true_w = torch.tensor([2,-3,4])
true_b = 4.2

features,lables = synthetic_data(true_w,true_b,1000)

#In[]
#调用现有API来读取数据

def load_array(data_arrays,batch_size,is_train = True):
    dataset = data.TensorDataset(*data_arrays) #TensorDataset是为通过沿第一维度索引张量来检索每个样本
    return data.DataLoader(dataset,batch_size,shuffle = is_train) #组合一个数据集和一个采样器，并在给定的数据集上提供一个iterable。

batch_size = 10
data_iter = load_array((features,lables),batch_size)

#In[]
#定义线性层 
net = nn.Sequential(nn.Linear(3,1)) #一个线性层 输入维度为3,输出维度为1

#初始化模型参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

#定义损失函数 
loss = nn.MSELoss(size_average=None)

#实例化optimizor
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)

#In[]
#训练部分
num_epochs = 3

for epoch in range(num_epochs):
    for X,y in data_iter: #从迭代器中获取X,y
        l = loss(net(X),y) #计算loss
        trainer.zero_grad() #将梯度清零
        l.backward() #求导
        trainer.step() #更新权重
    
    l = loss(net(features),lables) #计算本epoch最终loss
    print(f'epoch{epoch + 1},loss{l:f}')

# %%
