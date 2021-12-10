'''
GoogLeNet的基本结构是Inception block,
Inception_block block分为四路,每一路都不改变H和W的大小,通过1x1conv来降低计算复杂度

GoogLeNet用了5个stage,共9个Inception blocks
相比AlexNet,在一开始用更小的窗口和产生更多的channels
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


#先定义Inception block (V1版本)
#不是现成的结构必须自己定义class,进而定义前向传播
class Inception_block(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,*args,**kargs):
        '''
        in_channels:输入通道数
        c1~c4:分别对应4条路径的输出通道数
        '''
        super().__init__()

        #path1 一个1x1 conv
        self.path1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)

        #path2 1x1 conv and 3x3 conv with padding1 
        self.path2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.path2_2 = nn.Conv2d(in_channels,c2[1],kernel_size=3,padding=1)

        #path3 1x1 conv and 3x3 conv with padding2
        self.path3_1 = nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.path3_2 = nn.Conv2d(in_channels,c3[1],kernel_size=5,padding=2)

        #pqth4 3x3 Maxpool woth padding 1 and 1x1 conv
        self.path4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.path4_2 = nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,X):

        path1 = F.relu(self.path1_1(X))
        path2 = F.relu(self.path2_2(F.relu(self.path2_1(X))))
        path3 = F.relu(self.path3_2(F.relu(self.path3_1(X))))
        path4 = F.relu(self.path4_2(F.relu(self.path4_1(X))))

        return torch.cat((path1,path2,path3,path4),dim=1) #dim=1是channel维度,按channel维度拼接

#In[]
#逐一实现GLN的5个stage

#stage1
stage1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))




stage2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
stage3 = nn.Sequential(Inception_block(192, 64, (96, 128), (16, 32), 32),
                   Inception_block(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))



stage4 = nn.Sequential(Inception_block(480, 192, (96, 208), (16, 48), 64),
                   Inception_block(512, 160, (112, 224), (24, 64), 64),
                   Inception_block(512, 128, (128, 256), (24, 64), 64),
                   Inception_block(512, 112, (144, 288), (32, 64), 64),
                   Inception_block(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

stage5 = nn.Sequential(Inception_block(832, 256, (160, 320), (32, 128), 128),
                   Inception_block(832, 384, (192, 384), (48, 128), 128),#输出维度1024xH'xW'
                   nn.AdaptiveAvgPool2d((1,1)),#跟NiN一样,最后有个大平均池化层消去H和W维度 (-1,1024)
                   nn.Flatten())

net = nn.Sequential(stage1, stage2, stage3, stage4, stage5, nn.Linear(1024, 10))


#In[]
#训练阶段和之前相同 略