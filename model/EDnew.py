from __future__ import division
import torch
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F

# general libs
import cv2
from PIL import Image
import numpy as np
import math
import time
import os
import sys
import argparse
from components.utils import *


class ConvEnc(nn.Module):
    def __init__(self, c_in, c_out, kernel, stride, padding):
        super(ConvEnc, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, stride, padding, bias=False), # 
            nn.InstanceNorm2d(c_out, affine=True, momentum=0),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class DeConv(nn.Module):
    def __init__(self, c_in, c_out, kernel = 3, scale = 2):
        super(DeConv, self).__init__()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=scale)
        self.same_padding   = nn.ReflectionPad2d(1)
        self.conv           = nn.Conv2d(c_in, c_out, kernel, 1, 0, bias= False)
        self.__weights_init__()

    def __weights_init__(self):
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input):
        h   = self.upsampling(input)
        h   = self.same_padding(h)
        h   = self.conv(h)
        return h

class ConvDec(nn.Module):
    def __init__(self, c_in, c_out, kernel):
        super(ConvDec, self).__init__()
        self.layer = nn.Sequential(
            DeConv(c_in, c_out, kernel), # 
            nn.InstanceNorm2d(c_out, affine=True, momentum=0),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, chn):
        super(Encoder, self).__init__()
        self.layer1 = ConvEnc(3, chn, 4, 2, 1)
        self.layer2 = ConvEnc(chn, 2*chn, 4, 2, 1)
        self.layer3 = ConvEnc(2*chn, 4*chn, 4, 2, 1)
        self.layer4 = ConvEnc(4*chn, 8*chn, 4, 2, 1)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        return y


class Decoder(nn.Module):
    def __init__(self, chn):
        super(Decoder, self).__init__()
        self.layer1 = ConvDec(8*chn, 8*chn, 3)
        self.layer2 = ConvDec(8*chn, 4*chn, 3)
        self.layer3 = ConvDec(4*chn, 2*chn, 3)
        self.layer4 = ConvDec(2*chn, chn, 3)
        self.layer5 = nn.Conv2d(chn, 3, 1, 1, 0)

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        return y