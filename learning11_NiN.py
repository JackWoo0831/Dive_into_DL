#In[]
'''
NiN:Net in Net
为了解决之前网络全连接层参数过多的问题,因此一个NiN的block
是卷积层后紧跟两个1x1的卷积层,相当于共享权重的FC层.
在每个block后,都跟一个最大池化层,(3x3,stride=2)
最后有一个大的平均池化层
'''

import torch 
from torch import nn 

from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import resize

import Common_functions


#In[]
def NiN_block(in_channels,out_channels,kernel_size=3,strides=2,padding=1):
    '''
    一个NiN block,包括1个kernel_size卷积层和2个1x1卷积层
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding),
        nn.ReLU(),

        nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1),
        nn.ReLU()
    )


#NiN网络,为输入224设计
net = nn.Sequential(
    NiN_block(1, 96, kernel_size=11, strides=4, padding=0),#(96,54,54)
    nn.MaxPool2d(3, stride=2),#(96,26,26)
    NiN_block(96, 256, kernel_size=5, strides=1, padding=2),#(256,26,26)
    nn.MaxPool2d(3, stride=2),#(256,13,13)
    NiN_block(256, 384, kernel_size=3, strides=1, padding=1),#(384,13,13)
    nn.MaxPool2d(3, stride=2),#(384,6,6)
    nn.Dropout(0.5),
    # 标签类别数是10
    NiN_block(384, 10, kernel_size=3, strides=1, padding=1),#(10,6,6)
    nn.AdaptiveAvgPool2d((1, 1)), #将后两个维度抹掉(变成1x1) (10,1,1)
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())


#In[]
#train!

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
