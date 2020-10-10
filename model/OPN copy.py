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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv12 = GatedConv2d(3, 64, kernel_size=5, stride=2, padding=2, activation=nn.SELU()) # 2
        self.conv2 = GatedConv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.SELU()) # 2
        self.conv23 = GatedConv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.SELU()) # 4
        self.conv3a = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.SELU()) # 4
        self.conv3b = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, activation=nn.SELU()) # 4
        self.conv3c = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, activation=nn.SELU()) # 4
        self.conv3d = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=8, dilation=8, activation=nn.SELU()) # 4
        

    def forward(self, x):
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.conv3d(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv3d = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=8, dilation=8, activation=nn.SELU()) # 4
        self.conv3c = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, activation=nn.SELU()) # 4
        self.conv3b = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, activation=nn.SELU()) # 4
        self.conv3a = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.SELU()) # 4
        self.conv32 = GatedConv2d(128, 64, kernel_size=3, stride=1, padding=1, activation=nn.SELU()) # 2
        self.conv2 = GatedConv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.SELU()) # 2
        self.conv21 = GatedConv2d(64, 3, kernel_size=5, stride=1, padding=2, activation=None) # 1

    def forward(self, x):
        x = self.conv3d(x)
        x = self.conv3c(x)
        x = self.conv3b(x)
        x = self.conv3a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv32(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv21(x)
        return x