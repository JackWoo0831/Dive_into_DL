#In[]
'''
LeNet,上世纪80年代的产物,最初为了手写识别设计
'''

import torch 
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss

from torch.utils import data
import torchvision
from torchvision import transforms

import Common_functions


'''
LeNet:
两个卷积层,两个池化层,三个线性层
假定为MNIST设计,输入为(batch_size,1,28,28)
'''
net = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),padding=2),nn.Sigmoid(), #输出:(6,28,28)
    nn.AvgPool2d(kernel_size=(2,2)), #不指定stride默认不重叠 输出(6,14,14)
    nn.Conv2d(6,16,kernel_size=(5,5)),nn.Sigmoid(),#输出(16,10,10)
    nn.AvgPool2d(kernel_size=(2,2)),#输出(16,5,5)
    nn.Flatten(),
    nn.Linear(16*5*5,120),nn.Sigmoid(),#
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10)
)




#In[]

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

Common_functions.train_device(net,train_iter,test_iter,lr=0.9,device=device)
# %%
