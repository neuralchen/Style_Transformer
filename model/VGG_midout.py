import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
import math


class VGG16_mid(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.slice1 = vgg[: 4]
        self.slice2 = vgg[4: 9]
        self.slice3 = vgg[9: 16]
        self.slice4 = vgg[16: 23]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1,h2,h3,h4

if __name__ == "__main__":
    vgg = vgg16(pretrained=True).features
    print(vgg[: 4])
    print(vgg[4: 9])
    print(vgg[9: 16])
    print(vgg[16: 23])