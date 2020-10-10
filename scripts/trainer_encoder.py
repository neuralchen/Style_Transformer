import os
import argparse
from PIL import Image
import numpy as np
from skimage import io, transform
from tqdm import tqdm
import joblib
import glob
import datetime
import itertools

import torch
import torch.nn.functional as F
from torchvision import transforms,datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model.OPN import Encoder, Clsout
from components.utils import *

        

class Trainer(object):
    def __init__(self, config):

        self.trans = transforms.Compose([
                transforms.RandomSizedCrop(512),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(config.train_data_dir, self.trans),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True, drop_last=True)

        os.makedirs(f'{config.save_dir}/{config.version}',exist_ok=True)
        self.model_state_dir = f'{config.save_dir}/model_state'
        self.code_dir = f'{config.save_dir}/{config.version}/code'

        os.makedirs(self.model_state_dir,exist_ok=True)
        os.makedirs(self.code_dir,exist_ok=True)

        self.encoder = Encoder().cuda()
        self.out = Clsout().cuda()

        self.config = config

    def train(self):

        optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), 
                                                     self.out.parameters()), 
                                     lr=self.config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()#torch.nn.MSELoss()]


        self.encoder.train()
        self.out.train()

        for e in range(1, self.config.epoch_size+1):
            print(f'Start {e} epoch')
            for i, (content, target)  in enumerate(self.train_loader):
                content = content.cuda()
                target = target.cuda()

                latent_feature = self.encoder(content)
                classification = self.out(latent_feature)

                loss = criterion(classification, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if i % self.config.log_interval == 0:
                    import GPUtil
                    GPUtil.showUtilization()
                    now = datetime.datetime.now()
                    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(otherStyleTime)
                    print('epoch: ',e,' iter: ',i)
                    print('loss:', loss.cpu().item())

                    self.encoder.eval()
                    self.out.eval()

                    pred = self.out(self.encoder(content))
                    pred = torch.argmax(pred, -1)
                    acc = torch.sum((pred==target).float())/target.shape[0]
                    print('accuracy :',acc.item())

                    torch.save({'encoder':self.encoder.state_dict(),
                                'out':self.out.state_dict()}, f'{self.model_state_dir}/epoch_{e}-iter_{i}.pth')    
                    self.encoder.train()
                    self.out.train()

                

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('/media/qiuting/Data/datasets/Place365', transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True,drop_last=True)
    data = list(enumerate(self.train_loader, 1))
    print(len(data))