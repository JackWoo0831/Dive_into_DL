#In[]
'''
本节笔记:
有时为了防止过拟合,有两种大体的途径:一是减少模型自身的复杂度,二是对权重的取值范围进行一种限制,正则化就是
加以限制的一种方式:在loss中加入罚项(penalty),可以将参数w从最优点向原点拉.
'''
from random import SystemRandom, shuffle
from sys import flags
import torch 
from torch import nn 
from torch.utils import data

import Common_functions

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

#将Tensor包装成数据集,并产生迭代对象
def load_array(data_array,batch_size,is_train = True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle = is_train)

#定义参数
n_train,n_test,num_input,batch_size = 20,100,200,5

true_w, true_b = torch.ones((num_input,1))*0.01 , 0.05

train_data = synthetic_data(true_w,true_b,n_train)
train_iter = load_array(train_data,batch_size)

test_data = synthetic_data(true_w,true_b,n_test)
test_iter = load_array(test_data,batch_size,is_train=False)

#初始化参数
def init_params():
    w = torch.normal(0,0.1,size=(num_input,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)

    return w,b

#定义L2 penalty
def L2_penalty(w):
    return torch.sum(w.T @ w) / 2



#In[]
#Train!

def net(X,w,b):
    return torch.matmul(X,w) + b

def train(lbda):
    '''
    input:lbda:罚项的系数
    '''
    w,b = init_params()
    num_epochs , lr = 10,0.01

    for epoch in range(num_epochs):
        for X,y in train_iter:
            with torch.enable_grad():
                l = Common_functions.squared_loss(net(X,w,b),y) + lbda * L2_penalty(w)

            l.sum().backward()

            Common_functions.sgd([w,b],lr,batch_size)

        print(f'epoch:{epoch},loss:{l.sum()}')
    
    print(f'the norm of w is {torch.norm(w).item()}')





# %%
