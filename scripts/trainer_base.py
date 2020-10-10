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

# class ValidationDataLoader(object):
#     """Dataset class for the Artworks dataset and content dataset."""
#     def __init__(self, content_image_dir,style_image_dir,
#                     selectedContent,selectedStyle,
#                     content_transform,style_transform,
#                     subffix='jpg', random_seed=1234):
#         """Initialize and preprocess the CelebA dataset."""
#         self.content_image_dir= content_image_dir
#         self.style_image_dir  = style_image_dir
#         self.content_transform= content_transform
#         self.style_transform  = style_transform
#         self.selectedContent  = selectedContent
#         self.selectedStyle    = selectedStyle
#         self.subffix            = subffix
#         self.content_dataset    = []
#         self.art_dataset        = []
#         self.random_seed= random_seed
#         self.__preprocess__()
#         self.num_images = len(self.content_dataset)
#         self.art_num    = len(self.art_dataset)

#     def __preprocess__(self):
#         """Preprocess the Artworks dataset."""
#         print("processing content images...")
#         for dir_item in self.selectedContent:
#             join_path = Path(self.content_image_dir,dir_item)#.replace('/','_'))
#             if join_path.exists():
#                 print("processing %s"%dir_item)
#                 images = join_path.glob('*.%s'%(self.subffix))
#                 for item in images:
#                     self.content_dataset.append(item)
#             else:
#                 print("%s dir does not exist!"%dir_item)
#         label_index = 0
#         print("processing style images...")
#         for class_item in self.selectedStyle:
#             images = Path(self.style_image_dir).glob('%s/*.%s'%(class_item, self.subffix))
#             for item in images:
#                 self.art_dataset.append([item, label_index])
#             label_index += 1
#         random.seed(self.random_seed)
#         random.shuffle(self.content_dataset)
#         random.shuffle(self.art_dataset)
#         # self.dataset = images
#         print('Finished preprocessing the Art Works dataset, total image number: %d...'%len(self.art_dataset))
#         print('Finished preprocessing the Content dataset, total image number: %d...'%len(self.content_dataset))

#     def __getitem__(self, index):
#         """Return one image and its corresponding attribute label."""
#         filename        = self.content_dataset[index]
#         image           = Image.open(filename)
#         content         = self.content_transform(image)
#         art_index       = random.randint(0,self.art_num-1)
#         filename,label  = self.art_dataset[art_index]
#         image           = Image.open(filename)
#         style           = self.style_transform(image)
#         return content,style,label

#     def __len__(self):
#         """Return the number of images."""
#         return self.num_images

class Trainer_Base(object):
    def __init__(self, config):
        self.trans = transforms.Compose([
                transforms.RandomResizedCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
        self.criterion = torch.nn.MSELoss(reduction='mean')

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
        loss_perc = []
        loss_cont = []
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
                    fea_c = self.encoder(c)
                    fea_s = self.encoder(s)

                    out_feature, _ = self.attention1(fea_c, fea_s)

                    stylized = self.decoder(out_feature)

                    c1,c2,c3,_ = self.D(c)
                    h1,h2,h3,_ = self.D(stylized)
                    s1,s2,s3,_ = self.D(s)

                    loss_content = self.criterion(c3,h3)
                    loss_perceptual = 0
                    for t in range(3):
                        loss_perceptual += self.criterion( gram_matrix(eval('s'+str(t+1))), gram_matrix(eval('h'+str(t+1))) )
                    loss_perc.append(self.config.content_weight*loss_content.cpu().item())
                    loss_cont.append(self.config.style_weight*loss_perceptual.cpu().item())
                outs.append(align_shape(stylized, c))
                styles.append(s)

            out = torch.cat(outs, 2)
            if i==0:
                style = torch.cat(styles, 2)
                tosave.append(style[0])
            tosave.append(out[0])

        return torch.cat(tosave,2), sum(loss_perc)/len(loss_perc), sum(loss_cont)/len(loss_cont)
