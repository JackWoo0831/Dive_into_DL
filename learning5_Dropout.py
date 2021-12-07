#In[]
'''
笔记:
Dropout的做法是对于一个输入x_i,以概率p将x_i置为0,以概率(1-p)将x_i放大至x_i/(1-p)
通常作用在FC层的输出上

问题:准确率很低...
'''

import torch 
from torch import nn
from torch.nn.modules.activation import Softmax 
import torchvision
from torchvision import transforms
from torch.utils import data

import Common_functions


#定义Dropout函数,将输入的张量按概率变为0
def dropout_layer(X,p):
    '''
    input:
    X:input tonsor
    p:probability of dropout
    output:
    new tensor X'
    '''
    assert p >= 0 and p <= 1

    if p == 0:
        return X
    elif p == 1:
        return torch.zeros(X.shape)
    else:
        mask = (torch.rand(X.shape)>p).float() #生成0,1间uniform, 大于p的置为1,因此p越大1越少, mask和X相乘后0越多

        return (mask * X) / (1 - p)

#In[]
#定义有两个隐藏层的网络

num_inputs, num_outputs, num_hidden1, num_hidden2 = 784,10,256,256

p1, p2 = 0.2,0.5 #两个Dropout层的概率

class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hidden1,num_hidden2,is_train = True):
        super(Net,self).__init__()

        self.num_inputs = num_inputs
        self.training = is_train

        #输入层-隐藏层1-隐藏层2-输出层
        self.lin1 = nn.Linear(num_inputs,num_hidden1)
        self.lin2 = nn.Linear(num_hidden1,num_hidden2)
        self.lin3 = nn.Linear(num_hidden2,num_outputs)
        self.ReLU = nn.ReLU()

    def forward(self,X):
        #结构:input-hidden layer-relu-dropout-hidden layer-relu-dropout-output layer
        H1 = self.ReLU(self.lin1(X.reshape((-1,self.num_inputs))))
        if self.training == True:
            H1 = dropout_layer(H1,p1)
        
        H2 = self.ReLU(self.lin2(H1))

        if self.training == True:
            H2 = dropout_layer(H2,p2)

        output = self.lin3(H2)

        return output



net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2) #实例化

#In[]
#Train!

num_epochs, lr, batch_size = 15,0.5,256

loss = nn.CrossEntropyLoss()
 

trans = transforms.ToTensor() #class,将PIL图片或numpy.ndarray格式转为tensor的类

mnist_train = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=True,transform=trans,download=False)
mnist_test = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=False,transform=trans,download=False)

trian_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)

updater = torch.optim.SGD(net.parameters(),lr)

#Common_functions.train(net,trian_iter,test_iter,loss,num_epochs,updater)


#In[]
'''简洁实现,运行上面必要的代码'''
net = nn.Sequential(
    nn.Flatten(),

    nn.Linear(num_inputs,num_hidden1),
    nn.ReLU(),
    nn.Dropout(p=p1),

    nn.Linear(num_hidden1,num_hidden2),
    nn.ReLU(),
    nn.Dropout(p=p2),

    nn.Linear(num_hidden2,num_outputs)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,0,0.1)

net.apply(init_weights)

Common_functions.train(net,trian_iter,test_iter,loss,num_epochs,updater)



# %%
