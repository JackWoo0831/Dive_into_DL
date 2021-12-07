'''记录一些常用的函数'''

import torch 
from torch import tensor,nn
import torchvision

import numpy 

#import cv2

#1.评估分类准确率函数

def accuracy(y_hat,y):
    '''
    input:
    y_hat:预测结果
    y:Ground truth
    output:预测正确的个数
    '''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: #y应该是矩阵,行为样本列为类别
        y_hat = y_hat.argmax(axis=1) #求使得行最大的索引也就是类别
    cmp = y_hat.type(y.dtype) == y #转化为相同类型后比较
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    '''
    定义累加器类
    '''
    def __init__(self,n):
        self.data = [0.0] * n
    
    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self,idx):

        return self.data[idx]

def evaluate_accuracy(net,data_iter):
    '''
    input:
    net:网络模型
    data_iter:数据迭代器,train_iter or test_iter
    output:
    将data_iter遍历完后模型的准确率
    '''
    if isinstance(net,torch.nn.Module):
        net.eval() #如果是torch的API,就进入评估模式

    #用前面定义的Accumulator
    metric = Accumulator(2) #metric为类,metric.data表示一个长度为2的list,第一个元素是正确个数,第二个元素为总数
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())


    return metric[0] / metric[1]



def evaluate_accuracy_device(net,data_iter,device=None):
    '''
    可指定device的评估准确率函数
    input:net:model data_iter
    device:
    output:
    accuracy
    '''
    if isinstance(net,torch.nn.Module):
        net.eval()
        if not device:#如果device是None
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)

    with torch.no_grad(): #不知道为何要多这个
        for X,y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X),y),y.numel())

    return metric[0] / metric[1]



        

#train一个epoch的函数
def train_epoch(net,train_iter,loss,updater):
    '''
    input:
    net:模型
    train_iter:训练集数据迭代器
    loss:损失函数
    updater:优化器
    output:
    loss,acc
    '''
    if isinstance(net,torch.nn.Module): #这是为了代码以后复用方便,判断net是否为torch.nn.Module类型
        net.train() #pytorch首先要.train一下

    
    metirc = Accumulator(3)

    for X,y in train_iter:
        y_hat = net(X)

        l = loss(y_hat,y)

        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad() #梯度清零
            l.backward() #loss求导
            updater.step() #更新参数
            #向累加器中加三个数,第一是损失函数乘以y的个数(应该是为了算损失函数,注意损失函数是个向量)
            #第二是正确个数,第三是总数
            metirc.add(float(l) * len(y),accuracy(y_hat,y),y.size().numel())


        else:
            #如果是自己写的模型
            l.sum().backward() #求和之后再求导
            updater(X.shape[0])

            #print(l.sum(),accuracy(y_hat,y),y.size().numel())
            metirc.add(float(l.sum()),accuracy(y_hat,y),y.size().numel())

        return metirc[0] / metirc[2] , metirc[1] / metirc[2]


#训练总函数
def train(net,train_iter,test_iter,loss,num_epochs,updater):
    '''
    net:模型
    train_iter:train dataset iter
    test_iter:test dataset iter
    loss:loss func
    num_epochs:number of epochs
    updater:optimizer
    '''
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)


        train_loss,train_acc = train_metrics

        print(f'train_loss:{train_loss};train_acc:{train_acc}')
        print(f'test_acc:{test_acc}')


def train_device(net,train_iter,test_iter,loss=None,num_epochs=10,lr=0.1,updater=None,device=None):
    '''
    可指定device的训练函数,比原来的train多了to_device操作

    还可以修改的地方:指定初始化方式

    '''

    #先定义权重初始化函数,Xavier初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)

    if not updater:
        updater = torch.optim.SGD(net.parameters(),lr=lr)

    if not loss:
        loss = nn.CrossEntropyLoss()

    #train!
    for _ in range(num_epochs):
        metirc = Accumulator(3)

        for i,(X,y) in enumerate(train_iter):
            updater.zero_grad()
            X,y = X.to(device),y.to(device)

            y_hat = net(X)

            l = loss(y_hat,y)

            l.backward()
            updater.step()

            metirc.add(float(l) * len(y),accuracy(y_hat,y),y.size().numel())

    
        train_loss,train_acc = metirc[0] / metirc[2] , metirc[1] / metirc[2]

        print(f'train_loss:{train_loss};train_acc:{train_acc}')
    

    test_acc = evaluate_accuracy_device(net,test_iter,device)


    print(f'test_acc:{test_acc}')

#2.optimizers

#定义SGD
def sgd(params,lr,batch_size):
    '''
    input:
    params:list,elment:torch.Tensor.weights and bias
    lr:learning rate
    batch_size.
    '''
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size #SGD:w_t = w_{t-1} - lr * \partial l / \partial w 
            #此处/batch_size是因为损失函数那里没有除
            param.grad.zero_()


#3.loss functions

#实现交叉熵损失函数

def cross_entropy(y_hat,y):
    '''
    input:
    y_hat:估计类别
    y:GT
    交叉熵损失:GT类别对应的估计的概率取负对数再求和,求和是对所有样本求和
    '''

    return -torch.log(y_hat[range(len(y_hat)),y])


def squared_loss(y_hat,y):
    '''
    均方损失
    input:
    y_hat:估计类别
    y:GT    
    '''

    return (y_hat - y.reshape(y_hat.shape))**2 / 2 #reshape是防止出现行列向量不匹配的情况

