'''
ResNet:为了解决大网络收敛到最佳点的问题.当网络加深后,反而可能会离最优解越来越远.
如果网络不断加深的过程中,不同大小的网络都是严格包含的关系,那么就可以离最优点越来越近,至少不会更差
因此ResNet基本公式是: y = f(x) + x 如果f(x)不起作用,也不会更差.
'''

#In[]

import torch 
from torch import nn 
from torch.nn import functional as F

from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import resize

import Common_functions

#In[]

#定义residual块
'''
分为含有1x1conv(在支路)和不含1x1conv两种
主路一般是3x3conv + BatchNrom + ReLU + 3x3conv + BatchNorm(也有许多其他的变种)

Residual Block一般是两种用法:高宽减半,通道数加倍(stride=2,num_channels=2*in_channels)
或者高宽和通道数不变(stride=1,num_channels=in_channels)
'''

class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1x1conv=False,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,num_channels,kernel_size=3,padding=1,stride=stride)  #自定义卷积层

        self.conv2 = nn.Conv2d(
            num_channels,num_channels,kernel_size=3,padding=1) #该层卷积维持通道数不变 size不变

        if use_1x1conv is True:
            self.conv3 = nn.Conv2d(in_channels,num_channels,kernel_size=1,stride=stride) #跟第一层conv一样的stride 维持size相同
        else:
            self.conv3 = None
        
        #两个BatchNorm层
        self.bn1 = nn.BatchNorm2d(num_features=num_channels)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)


    def forward(self,X):
        
        Y = F.relu(self.bn1(self.conv1(X)))

        Y = self.bn2(self.conv2(Y))

        if self.conv3 is not None:
            X = self.conv3(X)

        Y += X

        return F.relu(Y) 


#In[]

'''ResNet模型
ResNet模型和GLN很像,都是5个stage.
首先都是经过7x7 stride=2的卷积后进入主体,由于7x7卷积已经对尺寸做出了改变,因此
第一个残差块不改变size,后面逐步对高宽减半,维度加倍,最后用全局池化层展平后经过线性层.
'''

#定义好几个残差块组成的块
def resnet_block(in_channels,num_channels,num_residuals,first_block = False):
    
    blk = []
    for i in range(num_residuals):

        if i == 0 and not first_block:
            #如果不是接在7x7后面的块
            blk.append(Residual(in_channels,num_channels,use_1x1conv=True,stride=2)) #高宽减半维度加倍

        else:
            blk.append(Residual(num_channels,num_channels)) #shape不变

    
    return blk 


#5个stage

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 = nn.Sequential(*resnet_block(64,128,2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1,b2,b3,b4,b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),nn.Linear(512,10))


#In[]
#测试shape

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

        
#训练略
        

# %%
