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
import shutil

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model.VGG import Encoder
from model.VGG_midout import VGG16_mid
from model.Decoder import Decoder
from model.Fusion import *
from components.utils import *
from components.transformer import Transformer
from scripts.trainer_base import Trainer_Base

class Trainer(Trainer_Base):

    def train(self):
        self.create_model()
        optimizer = torch.optim.Adam(itertools.chain(self.attention1.parameters(), 
                                                    #  self.attention2.parameters(), 
                                                    #  self.decoder.parameters()
                                                     ), 
                                     lr=self.config.learning_rate)
        criterion = torch.nn.L1Loss()
        criterion_p = torch.nn.MSELoss(reduction='mean')
        styles = iter(self.style_loader)

        self.encoder.eval()
        self.decoder.eval()
        for e in range(1, self.config.epoch_size+1):
            for i, (content, target)  in enumerate(self.train_loader):
                try:
                    style, target = next(styles)
                except:
                    styles = iter(self.style_loader)
                    style, target = next(styles)

                content = content.cuda()
                style = style.cuda()

                fea_c = self.encoder(content)
                fea_s = self.encoder(style)
                
                out_feature, attention_map = self.attention1(fea_c, fea_s)
                # out_feature, attention_map = self.attention2(out_feature, fea_s)
                rec, _ = self.attention1(fea_s, fea_s)
                out_content = self.decoder(out_feature)
                
                c1,c2,c3,_ = self.D(content)
                h1,h2,h3,_ = self.D(out_content)
                s1,s2,s3,_ = self.D(style)

                _,c,h,w = h3.shape
                loss_content = criterion(c1, h1)
                loss_perceptual = 0
                for t in range(3):
                    loss_perceptual += criterion( gram_matrix(eval('s'+str(t+1))), gram_matrix(eval('h'+str(t+1))) )
                loss = loss_content*self.config.content_weight + loss_perceptual*self.config.style_weight

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % self.config.log_interval == 0:
                    now = datetime.datetime.now()
                    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(otherStyleTime)
                    print('epoch: ',e,' iter: ',i)
                    print('content loss:', self.config.content_weight*loss_content.cpu().item())
                    print('perceptual loss:', self.config.style_weight*loss_perceptual.cpu().item())
                    print('attention scartters: ', torch.std(attention_map.argmax(-1).float(), 1).mean().cpu())
                    print(attention_map.shape)


                    # self.attention1.hard = True
                    self.attention1.eval()

                    tosave = self.eval()
                    save_image(denorm(tosave), self.image_dir+'/epoch_{}-iter_{}.png'.format(e,i))
                    print("image saved to " + self.image_dir + '/epoch_{}-iter_{}.png'.format(e,i))

                    # self.attention1.hard = False
                    self.attention1.train()

                    torch.save({'layer1':self.attention1.state_dict()
                                # 'layer2':self.attention2.state_dict()
                                }, f'{self.model_state_dir}/epoch_{e}-iter_{i}.pth') 
