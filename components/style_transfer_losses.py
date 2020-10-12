#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: style_transfer_losses.py
# Created Date: Sunday October 11th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 11th October 2020 10:24:56 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch
from torch import nn

class TransformLossBlock(nn.Module):
    def __init__(self, k_size = 10):
        super().__init__()
        padding_size = int((k_size -1)/2)
        # self.padding = nn.ReplicationPad2d(padding_size)
        self.pool = nn.AvgPool2d(k_size, stride=1,padding=padding_size)

    def forward(self, input_image):
        # h = self.padding(input)
        out = self.pool(input_image)
        return out