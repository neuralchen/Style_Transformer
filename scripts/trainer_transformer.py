import os
import argparse
from PIL import Image
import numpy as np
from skimage import io, transform
from tqdm import tqdm
import joblib
import glob
import datetime
import random
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
from components.dataset import HDR_LDR

        

class Trainer(object):
    def __init__(self, config):
        content_trans = transforms.Compose([
                transforms.Resize(config.image_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.train_loader = torch.utils.data.DataLoader(
            HDR_LDR(config.ldr_dir, config.hdr_dir, content_trans),
            batch_size=1, shuffle=True,
            num_workers=config.workers, pin_memory=True, drop_last=True)

        os.makedirs(f'{config.save_dir}/{config.version}',exist_ok=True)

        self.loss_dir = f'{config.save_dir}/{config.version}/loss'
        self.model_state_dir = f'{config.save_dir}/{config.version}/model_state'
        self.image_dir = f'{config.save_dir}/{config.version}/image'
        self.psnr_dir = f'{config.save_dir}/{config.version}/psnr'
        self.code_dir = f'{config.save_dir}/{config.version}/code'

        os.makedirs(self.loss_dir,exist_ok=True)
        os.makedirs(self.model_state_dir,exist_ok=True)
        os.makedirs(self.image_dir,exist_ok=True)
        os.makedirs(self.psnr_dir,exist_ok=True)
        os.makedirs(self.code_dir,exist_ok=True)

        script_name = 'trainer_' + config.script_name + '.py'
        shutil.copyfile(os.path.join('scripts', script_name), os.path.join(self.code_dir, script_name))
        shutil.copyfile('components/transformer.py', os.path.join(self.code_dir, 'transformer.py'))
        shutil.copyfile('model/Fusion.py', os.path.join(self.code_dir, 'Fusion.py'))

        self.encoder = Encoder().cuda()
        self.attention = Transformer(config.topk, True, False).cuda()
        self.decoder = Decoder().cuda()

        self.decoder.load_state_dict(torch.load("./hdr_decoder.pth"))
        
        self.config = config

    def train(self):

        optimizer = torch.optim.Adam(self.attention.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.L1Loss()#torch.nn.MSELoss()

        self.decoder.eval()
        self.attention.eval()
        self.encoder.eval()

        for e in range(1, self.config.epoch_size+1):
            print(f'Start {e} epoch')
            psnr_list = []
            # for i, (content, target)  in tqdm(enumerate(self.train_loader, 1)):
            for i, (ldr, hdr, ref)  in enumerate(self.train_loader):
                ### data prepare
                ldr = ldr.cuda()
                hdr = hdr.cuda()

                ldrs,hdrs = patch_crop(ldr, True),patch_crop(hdr, True)
                flag = random.randint(0,3)

                ldr_in = ldrs[flag]
                hdr_in = [hdrs[i] for i in range(4) if i!=flag]
                target = hdrs[flag]

                ### encode
                fea_ldr = self.encoder(ldr_in)
                fea_hdr = [self.encoder(hdr_i) for hdr_i in hdr_in]
                
                ### attention swap
                out_feature, attention_map = self.attention(fea_ldr, fea_hdr)
                target_feature = self.encoder(target)

                ### decode
                out_image = self.decoder(out_feature)
                out_image = align_shape(out_image, target)

                loss = criterion(out_image, target)*0.1
                loss_fea = criterion(out_feature, target_feature)*10
                loss = loss + loss_fea

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if i % self.config.log_interval == 0:
                    now = datetime.datetime.now()
                    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(otherStyleTime)
                    print('epoch: ',e,' iter: ',i)
                    print('loss:', loss.cpu().item())
                    print('loss_fea:', loss_fea.cpu().item())

                    self.attention.hard = True
                    self.attention.eval()
                    with torch.no_grad():
                        out_feature, _ = self.attention(fea_ldr, fea_hdr)
                        out_image = self.decoder(out_feature)
                        out_image = align_shape(out_image, target)

                    self.attention.hard = False
                    self.attention.train()

                    ldrs[flag] = out_image
                    print('attention scartters: ', torch.std(attention_map.argmax(-1).float(), 1).cpu())
                    print(attention_map.shape)

                    tosave = torch.cat([patch_crop(ldrs, False), hdr], -1)
                    save_image(denorm(tosave[0]).cpu(), self.image_dir+'/epoch_{}-iter_{}.png'.format(e,i))
                    print("image saved to " + self.image_dir + '/epoch_{}-iter_{}.png'.format(e,i))

                    torch.save(self.attention.state_dict(), f'{self.model_state_dir}/epoch_{e}-iter_{i}.pth') 

                
