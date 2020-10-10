       
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math
from components.utils import *


class Transformer(nn.Module):
    def __init__(self, head_num, hidden_size, N = 1, hard = False, drop = True):
        super(Transformer, self).__init__()
        self.attention = MultiHead(head_num, hidden_size, N, hard, drop)
        self.FFN = FeedForward(hidden_size, hidden_size)

    def forward(self, x, reference=None):
        if reference is not None:
            out,attention_map = self.attention(x, reference)
        else:
            out,attention_map = self.attention(x, x)
        out = self.FFN(out)
        return out,attention_map

class DeepEmbed(nn.Module):
    def __init__(self, indim=256, outdim=256):
        super().__init__()
        self.layer1 = nn.Linear(indim, outdim)
        self.activation = F.relu
        self.layer2 = nn.Linear(outdim, outdim)
        self.activation2 = F.relu

    def forward(self, x):
        y = self.layer1(x)
        y = self.activation(y)
        y = self.layer2(y)
        y = self.activation2(y)
        return y

class MultiHead(nn.Module):
    def __init__(self, head_num, hidden_size, n = 1, hard=False, dropout=True):
        super().__init__()
        self.hard = hard
        self.n = n
        self.dropout = dropout

        self.num_attention_heads = head_num
        self.attention_head_size = int(hidden_size / head_num)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = DeepEmbed(hidden_size, self.all_head_size)
        self.key = DeepEmbed(hidden_size, self.all_head_size)
        self.value = DeepEmbed(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.5)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, content, reference, attention_mask=None):
        b,c,w,h = content.size()
        content_states = content.view(b,c,-1).transpose(-1,-2)  ### b * wh * c
        reference_states = reference.view(b,c,-1).transpose(-1,-2)  ### b * wh * c

        mixed_query_layer = self.query(content_states)
        mixed_key_layer = self.key(reference_states)
        mixed_value_layer = self.value(reference_states)  ### b * wh * dim

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  ### b * head * wh * d

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        if self.hard:
            mask = torch.topk(attention_scores, k = self.n, dim = 1)[0]
            mask = mask[:, -1].unsqueeze(1).expand_as(attention_scores)
            mask = torch.lt(attention_scores, mask).detach()

            attention_scores = attention_scores.masked_fill(mask, -1e18)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if self.dropout and not self.hard:
            attention_probs = self.dropout(attention_probs)  ### b * head * wh_q * wh_k
        context_layer = torch.matmul(attention_probs, value_layer)  ### b * head * wh_q * d

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  ### b * wh_q * head * wh_k
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape).permute(0, 2, 1).contiguous()  ### b * dim * wh_q

        outputs = context_layer.view(b,c,w,h)
        return outputs, attention_probs


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
        