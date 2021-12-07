#In[]
'''从0实现'''
import torch
from torch import tensor 
import torchvision 
from torch.utils import data 
from torchvision import transforms

import Common_functions


trans = transforms.ToTensor() #class,将PIL图片或numpy.ndarray格式转为tensor的类

#下载MNIST数据集
mnist_train = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=True,transform=trans,download=False)
mnist_test = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=False,transform=trans,download=False)


#In[]
#读取数据
batch_size = 256

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)

#将每个图片拉长成28x28=784
num_inputs = 784
#10类
num_outputs = 10

#初始化权重和偏置
w = torch.normal(0,0.01,(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)


#定义softmax
#softmax(X)_{i,j} = exp{X_{i,j}} / \sum_k{X_{i,k}}
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp / partition #对应元素相除


#定义网络,有两层,神经元层加softamx层
def net(X):
    return softmax(torch.matmul(X.reshape( (-1,w.shape[0]) ),w) + b)

#实现交叉熵损失函数
#交叉熵损失:GT类别对应的估计的概率取负对数再求和,求和是对所有样本求和
def cross_entropy(y_hat,y):

    #print(y_hat.shape,y.shape)
    return -torch.log(y_hat[range(len(y_hat)),y])

#In[]

#定义SGD
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size #SGD:w_t = w_{t-1} - lr * \partial l / \partial w 
            #此处/batch_size是因为损失函数那里没有除
            param.grad.zero_()

lr = 0.1

#updater就是SGD 
def updater(batch_size):
    return sgd([w,b],lr,batch_size)

#In[]
#Train!

num_epochs = 10

Common_functions.train(net,train_iter,test_iter,cross_entropy,num_epochs,updater)


#In[]
'''API实现'''
import torch 
import torch.nn as nn

batch_size = 256

trans = transforms.ToTensor() #class,将PIL图片或numpy.ndarray格式转为tensor的类

#下载MNIST数据集
mnist_train = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=True,transform=trans,download=False)
mnist_test = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=False,transform=trans,download=False)

#读取数据
batch_size = 256

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)

#In[]
#定义结构
net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(current_layer): #对当前层进行权重初始化
    if type(current_layer) == nn.Linear:
        nn.init.normal_(current_layer.weight,mean=0,std=0.01)

net.apply(init_weights) #apply到net上

#定义loss
loss = nn.CrossEntropyLoss()

#定义trainer
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs = 10

#!!之前实现的train 跑代码别忘了run一下之前的
Common_functions.train(net,train_iter,test_iter,loss,num_epochs,trainer)



# %%

# %%
