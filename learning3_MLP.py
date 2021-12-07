#In[]
'''MLP 从0开始'''
import torch 
from torch import nn 
import torchvision
from torchvision import transforms
from torch.utils import data

import Common_functions


trans = transforms.ToTensor() #class,将PIL图片或numpy.ndarray格式转为tensor的类

mnist_train = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=True,transform=trans,download=False)
mnist_test = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=False,transform=trans,download=False)


#In[]

batch_size = 256

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)

#网络结构是第一层:(-1,784) + ReLU + (784,256) + ReLU + (256,10) + softmax
num_inputs, num_outputs, num_hiddens = 784,10,256  

W1 = torch.randn(num_inputs,num_hiddens,requires_grad=True)
b1 = torch.zeros(num_hiddens,requires_grad=True)

W2 = torch.randn(num_hiddens,num_outputs,requires_grad=True)
b2 = torch.zeros(num_outputs,requires_grad=True)

params = [W1,b1,W2,b2]


#定义ReLU
def ReLU(X):
    Zeros = torch.zeros_like(X)

    return torch.max(Zeros,X)  #逐个元素比较

#定义网络
def net(X):
    X = X.reshape((-1,num_inputs)) #X:(bacth_size,784)
    H = ReLU((X @ W1) + b1)
    return ((H @ W2) + b2)

loss = nn.CrossEntropyLoss()

#In[]
#Train!

num_epochs , lr = 10,0.2

updater = torch.optim.SGD(params,lr)

#train函数修好放这里
Common_functions.train(net,train_iter,test_iter,loss,num_epochs,updater)

#In[]
'''简洁方式'''
import torch 
from torch import nn 

batch_size = 256

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)

net = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,0,0.1)

net.apply(init_weights)

batch_size,lr,num_epochs = 256,0.1,10

loss = nn.CrossEntropyLoss()

updater = torch.optim.SGD(net.parameters(),lr)

#要加载好train_iter和test_iter


Common_functions.train(net,train_iter,test_iter,loss,num_epochs,updater)

# %%
