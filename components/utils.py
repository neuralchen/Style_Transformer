import os
import argparse
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from skimage import io, transform
from PIL import Image



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

def align_shape(tensor, target):
    if tensor.shape[-1] < target.shape[-1]:
        tensor = torch.cat([tensor, tensor[:,:,:,tensor.shape[-1]-target.shape[-1]:].detach()],-1)
    else:
        tensor = tensor[:,:,:,:target.shape[-1]]
    if tensor.shape[-2] < target.shape[-2]:
        tensor = torch.cat([tensor, tensor[:,:,tensor.shape[-2]-target.shape[-2]:,:].detach()],-2)
    else:
        tensor = tensor[:,:,:target.shape[-2],:]
    return tensor

def patch_crop(tensor, in_):
    if in_:
        b,c,w,h = tensor.shape
        out = []
        mid_w = w//2
        mid_h = h//2
        out.append(tensor[:,:,:mid_w,:mid_h])
        out.append(tensor[:,:,mid_w:,:mid_h])
        out.append(tensor[:,:,:mid_w,mid_h:])
        out.append(tensor[:,:,mid_w:,mid_h:])
        return out
    else:
        out1,out2 = torch.cat(tensor[:2],-2),torch.cat(tensor[2:],-2)
        out = torch.cat([out1,out2], -1)
        return out

def init_He(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def gram_matrix(tensor):
    b,c,h,w = tensor.shape
    return torch.einsum('bxmn, bymn -> bxy', tensor, tensor)/(c*h*w)

def gram_matrix_cxh(tensor):
    (b, ch, h, w) = tensor.size()
    features = tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def torch_cosine_similarity(a,b):
    m = a.unsqueeze(-1)
    n = b.unsqueeze(0).permute(0,2,1)
    return torch.cosine_similarity(m,n)

def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).cuda()
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).cuda()
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def showimg(img):
    img = img.clamp(min=0, max=1)
    img = img.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()

def four2one(A,B,C,D):
    length = len(A.shape)
    up = torch.cat([A,B], length-2)
    down = torch.cat([C,D], length-2)
    return torch.cat([up,down], length-1)

def vector_swap(vector):
    tmp = np.array(vector[1])
    vector[1] = np.array(vector[0])
    vector[0] =np.array(tmp)
    return 0

def quick_sort(alist, blist,clist, start, end):
    # print(alist.shape,blist.shape,clist.shape)
    if start >= end:
        return
    mid = alist[start]
    bmid = blist[start]
    cmid = clist[start]
    left = start
    right = end
    while left < right:
        while left < right and alist[right] >= mid:
            right -= 1
        alist[left] = alist[right]
        blist[left] = blist[right]
        clist[left] = clist[right]
        while left < right and alist[left] < mid:
            left += 1
        alist[right] = alist[left]
        blist[right] = blist[left]
        clist[right] = clist[left]
    alist[left] = mid
    blist[left] = bmid
    clist[left] = cmid
    quick_sort(alist, blist, clist, start, left-1)
    quick_sort(alist, blist, clist,left+1, end)
    
def euclidean_dis(x,y):
    X=np.vstack([x,y])
    print(X.shape)
    sk=np.var(X,axis=0,ddof=1)
    print(sk.shape)
    return np.sqrt(((x - y) ** 2/sk).sum(axis=1))

def cos(x,y):
    dis = 1 - np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return dis

def PSNR(target, ref):  
    target_data = np.array(target)
    #target_data = target_data[0,scale:-scale,scale:-scale]
    ref_data = np.array(ref)
    #ref_data = ref_data[0,scale:-scale,scale:-scale]
    
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)

def PSNR2(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))