import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg19
import math
from components.utils import *

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation=None):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        init_He(self)
        self.activation = activation

    def forward(self, input):
        # O = act(Feature) * sig(Gating)
        feature = self.input_conv(input)
        if self.activation:
            feature = self.activation(feature)
        gating = F.sigmoid(self.gating_conv(input))
        return feature * gating

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = GatedConv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.key(x)

class Encoder(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        vgg = vgg19(pretrained=pretrain).features
        self.slice1 = vgg[: 5]
        self.slice2 = vgg[5: 10]
        self.slice3 = vgg[10: 19]
        self.slice4 = vgg[19: 28]
        # self.slice5 = vgg[28: 32]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        # h5 = self.slice5(h4)
        return h4

if __name__ == '__main__':
    vgg = vgg19(pretrained=False).features
    print(vgg[:32])