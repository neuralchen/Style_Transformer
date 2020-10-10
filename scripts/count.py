import os
import argparse
from PIL import Image
import numpy as np
from skimage import io, transform
from tqdm import tqdm
import joblib
import glob

import torch
import torch.nn.functional as F
from torchvision import transforms,datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
        


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/media/qiuting/Data/datasets/Place365', transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])),
            batch_size=8, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)
    for i, (content, target)  in tqdm(enumerate(train_loader, 1)):
        if i%1000 == 0:
            print(i)