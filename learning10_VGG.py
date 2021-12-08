'''
VGG是将卷积层进行模块化,便于更改和调整,也让网络看上去更有条理.
'''
#In[]
import torch 
from torch import nn 

from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import resize

import Common_functions

#定义一个VGG的block
def vgg_block(conv_num,in_channels,out_channels,kernel_size=3):
    '''
    conv_num:卷积层数量
    in_channels:输入通道数
    out_channels:输出通道数

    3x3卷积,padding1,stride1是不改变大小的,若采用2x2池化,每走完一个block H和W维度就减半
    '''
    layers = []

    for _ in range(conv_num):
        layers.append(nn.Conv2d(
            in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=1 
            ))

        layers.append(nn.ReLU())

        in_channels = out_channels #维数要衔接

    layers.append(nn.MaxPool2d(kernel_size=2))

    return nn.Sequential(*layers) #将layers中全体对象返回


#设计一个VGG11网络
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)) #(conv_numbers,output_channels)

def VGG(conv_arch,in_channels=1):
    '''
    conv_arch:tuple or list:(conv_nums,output_channels) x n
    '''
    blocks = []

    for conv_nums,out_channels in conv_arch:
        blocks.append(vgg_block(conv_nums,in_channels,out_channels))

        in_channels = out_channels

    return nn.Sequential(
        *blocks,
        #下面和AlexNet相同
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7,4096), #如前所述,每经过一个block就减半
        nn.ReLU(),nn.Dropout(p=0.5),

        nn.Linear(4096,4096),
        nn.ReLU(),nn.Dropout(p=0.5),

        nn.Linear(4096,10)

        
    )

net = VGG(conv_arch)


#In[]
#训练 
#对于小数据集,采用低复杂度的模型,将conv_arch里的维度减小

ratio = 4
conv_arch_ = [(pair[0],pair[1] // 4) for pair in conv_arch]

net_ = VGG(conv_arch_)


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
