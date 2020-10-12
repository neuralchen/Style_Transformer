import os
import argparse
from PIL import Image
import numpy as np
from skimage import io, transform
from tqdm import tqdm
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
from components.reporter import Reporter
from components.style_transfer_losses import TransformLossBlock

class Trainer_Base(object):
    def __init__(self, config):
        self.trans = transforms.Compose([
                transforms.RandomResizedCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.L1_loss   = torch.nn.L1Loss()
        self.transform_loss = TransformLossBlock().cuda()

        self.version_dir = f'{config.save_dir}/{config.version}'
        self.model_state_dir = f'{config.save_dir}/{config.version}/model_state'
        self.image_dir = f'{config.save_dir}/{config.version}/image'
        self.code_dir = f'{config.save_dir}/{config.version}/code'
        os.makedirs(self.version_dir,exist_ok=True)
        os.makedirs(self.model_state_dir,exist_ok=True)
        os.makedirs(self.image_dir,exist_ok=True)
        os.makedirs(self.code_dir,exist_ok=True)

        if config.content == 'decoder':
            script_name = 'trainer_' + config.script_name + '.py'
            shutil.copyfile(os.path.join('scripts', script_name), os.path.join(self.code_dir, script_name))
            shutil.copyfile('components/transformer.py', os.path.join(self.code_dir, 'transformer.py'))
            shutil.copyfile('model/EDnew.py', os.path.join(self.code_dir, 'EDnew.py'))

        self.reporter = Reporter(f'{config.save_dir}/{config.version}/{config.version}')
        self.reporter.writeConfig(config)


        self.config = config

    def plot_loss_curve(self, losses):
        for key in losses.keys():
            plt.plot(range(len(losses[key])), losses[key], label=key)
        plt.xlabel('iteration')
        plt.title(f'learning curve')
        plt.legend()
        plt.savefig(f'{self.version_dir}/{self.config.version}_curve.png')
        plt.clf()

    def create_model(self):
        self.encoder = Encoder(True).cuda()
        self.decoder = Decoder(True, True).cuda()
        self.D = VGG16_mid().cuda()
        self.attention1 = Transformer(4,512,self.config.topk,True,False).cuda()
        # self.attention2 = Transformer(4,512,config.topk,True,False).cuda()
        # self.decoder.load_state_dict(torch.load("./decoder.pth"))

    def test(self):
        params = torch.load(f'{self.model_state_dir}/epoch_{self.config.test_epoch}-iter_{self.config.test_batch}.pth')
        self.attention1.load_state_dict(params['layer1'])
        self.attention1.eval()

        self.eval()

    def eval(self):
        test_dir = f'{self.config.save_dir}/{self.config.version}/test'
        os.makedirs(test_dir,exist_ok=True)

        test_styles = ['patchsize5','patchsize7',
                       'patchsize8','patchsize9',
                       'patchsize11','patchsize13']
        fix_contents = glob.glob(self.config.fix_data_dir+'/*.jpg')[:5]
        tosave = []
        loss_content_list = []
        loss_style_list = []
        for i,C in enumerate(fix_contents):
            c = Image.open(C)
            c = self.trans(c).cuda().unsqueeze(0)
            _,_,W,H = c.shape

            outs = [c]
            styles = [torch.ones(c.shape).cuda()]
            for S in test_styles:
                S_path = os.path.join('./style', S)
                s = Image.open(glob.glob(S_path + '/*.jpg')[0]).resize((H,W),0)
                s = self.trans(s).cuda().unsqueeze(0)

                with torch.no_grad():
                    fea_c   = self.encoder(c)
                    fea_s   = self.encoder(s)
                    
                    out_feature, _  = self.attention1(fea_s, fea_c)

                    identity_c,_    = self.attention1(fea_c, fea_c)
                    identity_s,_    = self.attention1(fea_s, fea_s)

                
                    out_content     = self.decoder(out_feature)

                    rec_s           = self.decoder(identity_s)
                    rec_c           = self.decoder(identity_c)

                    # _, _, c3, _     = self.D(content)
                    h1, h2, h3, h4  = self.D(out_content)
                    s1, s2, s3, s4  = self.D(s)

                    # loss_perce      = self.L1_loss(c3,h3)
                    # loss_transform  = self.criterion(self.transform_loss(out_content),self.transform_loss(content))
                    # loss_perce      = self.L1_loss(fea_c, result_feat) # style aware loss
                    loss_content    = self.criterion(out_content, c)
                    # loss_content = loss_content - criterion(image_c.mean(-1), stylized_c.mean(-1))
                    loss_content    = loss_content + self.criterion(rec_s, s) + self.criterion(rec_c, c)
                    # loss_content    = self.config.feature_weight * loss_perce + \
                    #                     self.config.transform_weight * loss_transform
                    loss_content    = self.config.feature_weight * loss_content
                    loss_style      = self.criterion(gram_matrix_cxh(s1), gram_matrix_cxh(h1)) + \
                                        self.criterion(gram_matrix_cxh(s2), gram_matrix_cxh(h2)) + \
                                        self.criterion(gram_matrix_cxh(s3), gram_matrix_cxh(h3)) + \
                                        self.criterion(gram_matrix_cxh(s4), gram_matrix_cxh(h4))
                    loss_content_list.append(loss_content.cpu().item())
                    loss_style_list.append(self.config.style_weight*loss_style.cpu().item())
                outs.append(align_shape(out_content, c))
                styles.append(s)

            out = torch.cat(outs, 2)
            if i==0:
                style = torch.cat(styles, 2)
                tosave.append(style[0])
            tosave.append(out[0])

        return torch.cat(tosave,2), sum(loss_content_list)/len(loss_content_list), sum(loss_style_list)/len(loss_style_list)
