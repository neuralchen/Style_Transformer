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
from components.transformer_deep import Transformer
from components.reporter import Reporter
from scripts.trainer_base import Trainer_Base

class Trainer(Trainer_Base):

    def create_model(self):
        self.encoder = Encoder(True).cuda()
        self.decoder = Decoder(True, True).cuda()
        self.D = VGG16_mid().cuda()
        self.attention1 = Transformer(4,512,self.config.topk,True,False).cuda()

    def train(self):
        self.create_model()

        optimizer = torch.optim.Adam(self.attention1.parameters(), 
                                     lr=self.config.learning_rate)
        optimizer2 = torch.optim.Adam(self.decoder.parameters(), 
                                     lr=self.config.learning_rate)

        criterion = torch.nn.L1Loss()
        criterion_p = torch.nn.MSELoss(reduction='mean')
        styles = iter(self.style_loader)

        self.encoder.eval()
        self.decoder.train()
        self.reporter.writeInfo("Start to train the model")
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

                loss_content = torch.norm(c3-h3,p=2)
                loss_perceptual = 0
                for t in range(3):
                    loss_perceptual += criterion( gram_matrix(eval('s'+str(t+1))), gram_matrix(eval('h'+str(t+1))) )
                loss = loss_content*self.config.content_weight + loss_perceptual*self.config.style_weight

                optimizer.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer2.step()

                if i % self.config.log_interval == 0:
                    now = datetime.datetime.now()
                    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(otherStyleTime)
                    print('epoch: ',e ,' iter: ',i)
                    print('attention scartters: ', torch.std(attention_map.argmax(-1).float(), 1).mean().cpu())
                    print(attention_map.shape)

                    # self.attention1.hard = True
                    self.attention1.eval()
                    self.decoder.eval()
                    tosave,perc,cont = self.eval()
                    save_image(denorm(tosave), self.image_dir+'/epoch_{}-iter_{}.png'.format(e,i))
                    print("image saved to " + self.image_dir + '/epoch_{}-iter_{}.png'.format(e,i))
                    print('content loss:', cont)
                    print('perceptual loss:', perc)

                    self.reporter.writeTrainLog(e,i,f'''
                        attention scartters: {torch.std(attention_map.argmax(-1).float(), 1).mean().cpu()}\n
                        content loss: {cont}\n
                        perceptual loss: {perc}
                    ''')
                    

                    # self.attention1.hard = False
                    self.attention1.train()
                    self.decoder.train()

                    torch.save({'layer1':self.attention1.state_dict(),
                                # 'layer2':self.attention2.state_dict(),
                                'decoder':self.decoder.state_dict()}, f'{self.model_state_dir}/epoch_{e}-iter_{i}.pth') 
