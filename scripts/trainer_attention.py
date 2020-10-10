import os
import argparse
from PIL import Image
import numpy as np
from skimage import io, transform
from tqdm import tqdm
import joblib
import glob
import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms,datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from model.VGG import Encoder,KeyEncoder
from model.VGG_midout import VGG16_mid
from model.Decoder import Decoder
from components.utils import *
from components.transformer import *
from components.wavelet import *

        

class Trainer(object):
    def __init__(self, config):

        content_trans = transforms.Compose([
                transforms.RandomSizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(config.train_data_dir, content_trans),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True,drop_last=True)

        self.trans = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        style_trans = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        self.loss_dir = f'{config.save_dir}/loss'
        self.model_state_dir = f'{config.save_dir}/model_state'
        self.image_dir = f'{config.save_dir}/image'
        self.psnr_dir = f'{config.save_dir}/psnr'

        if not os.path.exists(self.loss_dir):
            os.mkdir(self.loss_dir)
            os.mkdir(self.model_state_dir)
            os.mkdir(self.image_dir)
            os.mkdir(self.psnr_dir)

        self.encoder = Encoder().cuda()
        self.transformer = Attention().cuda()
        self.decoder = Decoder().cuda()

        self.wavepool = WavePool(256).cuda()

        self.decoder.load_state_dict(torch.load("./decoder.pth"))
        
        S_path = os.path.join(config.style_dir, str(config.S))
        style_images = glob.glob((S_path + '/*.jpg'))
        s = Image.open(style_images[0])
        s = s.resize((512,320),0)
        s = style_trans(s).cuda()
        self.style_image = s.unsqueeze(0)
        self.style_target = torch.stack([s for i in range(config.batch_size)],0)

        self.config = config

    def train(self):

        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.L1Loss()#torch.nn.MSELoss()
        criterion_p = torch.nn.MSELoss(reduction='mean')


        self.transformer.train()

        for e in range(1, self.config.epoch_size+1):
            print(f'Start {e} epoch')
            psnr_list = []
            # for i, (content, target)  in tqdm(enumerate(self.train_loader, 1)):
            for i, (content, target)  in enumerate(self.train_loader):
                content = content.cuda()

                content_feature = self.encoder(content)
                style_feature = self.encoder(self.style_target)
                # rec_content = self.decoder(content_feature)
                
                out_feature = self.transformer(content_feature, style_feature)

                loss = criterion_p(self.wavepool(out_feature)[0], self.wavepool(content_feature)[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if i % self.config.log_interval == 0:
                    now = datetime.datetime.now()
                    otherStyleTime = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(otherStyleTime)
                    print('epoch: ',e,' iter: ',i)
                    print('loss:', loss.cpu().item())

                    self.encoder.eval()
                    self.transformer.eval()
                    self.decoder.eval()

                    with torch.no_grad():
                        test_image = Image.open('test.jpg')
                        test_image = self.trans(test_image).unsqueeze(0).cuda()
                        content_feature = self.encoder(content)
                        style_feature = self.encoder(self.style_image)
                        out_feature = self.transformer(content_feature, style_feature)
                        out_content = self.decoder(out_feature)

                    self.transformer.train()

                    save_image(denorm(out_content), self.image_dir+'/epoch_{}-iter_{}.png'.format(e,i))
                    print("image saved to " + self.image_dir + '/epoch_{}-iter_{}.png'.format(e,i))

                    model_dicts = self.transformer.state_dict()
                    torch.save(model_dicts, f'{self.model_state_dir}/epoch_{e}-iter_{i}.pth') 

                

            # filename = 'psnr/e'+ str(e) + '.pkl'
            # joblib.dump(psnr_list,os.path.join(self.config.save_dir,filename))


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