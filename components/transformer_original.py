       
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math
from components.utils import *


class Transformer(nn.Module):
    def __init__(self, N = 1, show = False, hard = True, inchannel=256, head_channel = 256):
        super(Transformer, self).__init__()
        self.attention = MultiHead(N, show, hard, inchannel, head_channel)
        self.FFN = FeedForward(inchannel, head_channel, 3, 1, 1)

    def forward(self, x, reference=None):
        if reference is not None:
            out,attention_map = self.attention(x, reference)
        else:
            out,attention_map = self.attention(x, [x])
        out = self.FFN(out+x)
        return out+x,attention_map

class MultiHead(nn.Module):
    def __init__(self, N = 1, show = False, hard = True, inchannel=256, head_channel = 256):
        super(MultiHead, self).__init__()
        self.K_emb = nn.Linear(inchannel, head_channel*3, bias=False)
        self.Q_emb = nn.Linear(inchannel, head_channel*3, bias=False)
        self.V_emb = nn.Linear(inchannel, head_channel*3, bias=False)
        self.out_emb = nn.Linear(head_channel*3, inchannel, bias=False)
        self.n = N
        self.show = show
        self.hard = hard

    def forward(self, content, reference):
        B,C,W,H = content.shape

        content = content.view(B,C,-1).transpose(2,1) ### b * len_c * c
        references = [ref.view(B,C,-1) for ref in reference]
        reference = torch.cat(references, -1).transpose(2,1) ### b * len_r * c

        Q = self.Q_emb(content) ### b * len_c * Dk
        K, V = self.K_emb(reference), self.V_emb(reference) ### b * len_r * Dk, b * len_r * Dv
        _,_,Dk = K.shape
        _,_,Dv = V.shape

        out = []
        ps = []
        for b in range(B):
            reads = []
            for i in range(3):
                Q_b = Q[b,:,C*i:C*i+C] # len_c * Dk
                K_b = K[b,:,C*i:C*i+C] # len_r * Dk
                V_b = V[b,:,C*i:C*i+C] # len_r * Dv

                try:
                    p = torch.mm(Q_b, K_b.T) # len_c * len_r
                except:
                    p = torch.mm(Q_b, K_b.t()) # len_c * len_r
                p = p/math.sqrt(Dk) 

                if self.hard:
                    mask = torch.topk(p, k = self.n, dim = 1)[0]
                    mask = mask[:, -1].unsqueeze(1).expand_as(p)
                    mask = torch.lt(p, mask).detach()

                    p = p.masked_fill(mask, -1e18)

                p = F.softmax(p, dim=1)
                ps.append(p)

                read = torch.mm(p, V_b) # len_r * Dv
                reads.append(read)
            read = torch.cat(reads, 1)
            read = self.out_emb(read)
            try:
                out.append(read.T.view(C, W, H))
            except:
                out.append(read.t().view(C, W, H))

        if self.show:
            return torch.stack(out, 0), torch.stack(ps, 0)
        else:
            return torch.stack(out, 0)



class FeedForward(nn.Module):
    def __init__(self, inchannel=256, outchannel=256, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.SELU()
        self.conv2 = nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        y = self.conv1(x)
        y = self.activation(y)
        y = self.conv2(y)
        return y
        