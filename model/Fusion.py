import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg19
import math


class Refine(nn.Module):
    def __init__(self,channels=256,kernel=3,stride=1,padding=1,bias=False):
        super(Refine, self).__init__()

        self.Att = FusionAttention(channels)
        self.Res = FusionResBlock(channels,kernel,stride,padding,bias)
        
    def forward(self,x1,x2):
        att_out = self.Att(x1,x2)
        y = self.Res(att_out,x2)
        return y

class FusionResBlock(nn.Module):
    def __init__(self,channels=256,kernel=3,stride=1,padding=1,bias=False):
        super(FusionResBlock, self).__init__()
        
        self.c = nn.Conv2d(channels*2, channels, kernel, stride, padding)
        self.b = nn.InstanceNorm2d(channels,affine=True)

    def forward(self,x1,x2):
        x = torch.cat([x1,x2], 1)
        h = self.c(x)
        h = self.b(h)
        y = x1 + h
        return y

class FusionAttention(nn.Module):
    def __init__(self, channels=256):
        super(FusionAttention, self).__init__()
        self.Embedding = nn.Linear(channels*2, channels*5, bias=False)

    def forward(self,x1,x2):
        B,C,W,H = x1.shape
        x = torch.cat([x1,x2], 1)
        x = x.view(B,2*C,-1).transpose(2,1) ### b * len_c * c

        embeddings = self.Embedding(x)

        Q = embeddings[:,:,:C*2] ### b * len_c * Dk
        K, V = embeddings[:,:,C*2:C*4], embeddings[:,:,C*4:] ### b * len_r * Dk, b * len_r * Dv
        Dk = C*2
        Dv = C

        out = []
        ps = []
        for b in range(B):
            Q_b = Q[b,:] # len_c * Dk
            K_b = K[b,:] # len_r * Dk
            V_b = V[b,:] # len_r * Dv

            try:
                p = torch.mm(Q_b, K_b.T) # len_c * len_r
            except:
                p = torch.mm(Q_b, K_b.t()) # len_c * len_r
            p = p/math.sqrt(Dk) 
            p = F.softmax(p, dim=1)

            ps.append(p)

            read = torch.mm(p, V_b) # len_r * Dv
            try:
                out.append(read.T.view(Dv, W, H))
            except:
                out.append(read.t().view(Dv, W, H))

        return torch.stack(out, 0)
