'''
AlexNet
'''
#In[]
import torch 
from torch import nn

from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import resize

import d2l

import Common_functions

#实现AlexNet
#为FashionMNIST设计,输入channel为1,但假定大小为224,224(28太小了)
#(最初的AlexNet输入为(3,224,224))

net = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=1),nn.ReLU(),#(96,54,54)
    nn.MaxPool2d(kernel_size=3,stride=2),#(96,26,26)

    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),#(256,26,26)
    nn.MaxPool2d(kernel_size=3,stride=2),#(256,12,12)

    nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),#(384,12,12)
    nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),#(384,12,12)
    nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),#(256,12,12)
    nn.MaxPool2d(kernel_size=3,stride=2),#(256,5,5)

    nn.Flatten(6400,4096),#(256*5*5->4096)
    nn.ReLU(),nn.Dropout(p=0.5),

    nn.Flatten(4096,4096),#(4096)
    nn.ReLU(),nn.Dropout(p=0.5),

    nn.Linear(4096,10)#(10)
)


#In[]

#读取数据

#要将图片resize成224,224
#参考https://zhuanlan.zhihu.com/p/91477545
trans = transforms.Compose(
    [transforms.Resize(size=(224,224)),
    transforms.ToTensor()]
)

mnist_train = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=True,transform=trans,download=False)
mnist_test = torchvision.datasets.FashionMNIST('datasets/FashionMnist/',train=False,transform=trans,download=False)

batch_size = 256

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter = data.DataLoader(mnist_test,batch_size,shuffle=True)



#train!
lr,num_epoch = 0.01,10

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

Common_functions.train_device(net,train_iter,test_iter,lr=lr,device=device)

# %%
