import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg19
import math


class GenResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,activation=F.selu):
        super(GenResBlock, self).__init__()

        self.activation = activation
        if h_ch is None:
            h_ch = out_ch
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        self.b1 = nn.InstanceNorm2d(in_ch,affine=True)
        self.b2 = nn.InstanceNorm2d(h_ch,affine=True)

    def forward(self, x):
        return x + self.residual(x)

    def residual(self, x):
        h = self.b1(x)
        h = self.activation(h)
        h = self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        return self.c2(h)


class RC(nn.Module):
    #A wrapper of ReflectionPad2d and Conv2d
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1,stride=1, bias=True, activated=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return torch.tanh(h)
        else:
            return h


class DeCov(nn.Module):
    def __init__(self,in_channels,out_channels,factor=2,kernel_size=3,stride=1,bias=False):
        super(DeCov, self).__init__()
        padding_size=(kernel_size-1)//2
        self.dconv=nn.Sequential(
            nn.Upsample(scale_factor=factor),
            nn.ReflectionPad2d((padding_size)),
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,bias=bias),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.SELU()
        )
        
    def forward(self,x):
        y=self.dconv(x)
        return y


class Cov(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,bias=False):
        super(Cov, self).__init__()
        padding_size=(kernel_size-1)//2
        self.conv=nn.Sequential(
                    nn.ReflectionPad2d((padding_size)),
                    nn.Conv2d(in_channels,out_channels,kernel_size,stride,bias=bias),
                    nn.InstanceNorm2d(out_channels,affine=True),
                    nn.SELU()
                    )

    def forward(self,x):
        x=self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, bias=True, activated=False):
        super().__init__()
        self.layer1 = DeCov(512, 512, 2)
        self.layer2 = nn.Sequential(
            Cov(512,512),
            Cov(512,512),
            Cov(512,512)
        )
        self.layer3 = DeCov(512, 256, 2)
        self.layer4 = Cov(256,256)
        self.layer5 = DeCov(256, 128, 2)
        self.layer6 = Cov(128,128)
        self.layer7 = DeCov(128, 64, 2)
        self.layer8 = Cov(64,64)
        self.layer9 = DeCov(64, 3, 1)
        
        self.rc = RC(3,3,bias=bias, activated=activated)

    def forward(self, features):
        h = self.layer1(features)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)
        h = self.layer7(h)
        h = self.layer8(h)
        h = self.layer9(h)
        h = self.rc(h)
        return h