import  os
import  time
import  datetime

import  torch
import  torch.nn as nn
import  numpy as np
import prettytable as pt
from    components.utils import *
import  torch.utils.data as data
from    torch.backends import cudnn
from    components.utils import *
from    torchvision.utils import save_image
from    model.VGG import *
from    model.Decoder import *
import  glob
from components.AAB import *
from components.wavelet import *

trans = transforms.Compose([transforms.RandomSizedCrop(654),
                            transforms.ToTensor(),
                            normalize])

class Tester(object):
    def __init__(self, config):

        self.content_images = glob.glob((config.exp_content_dir + '/*/*.jpg')) #+ '_resized/*'))

    
        self.encoder = Encoder().cuda()
        self.decoder = Decoder()
        self.keyencoder = KeyEncoder().cuda()

        self.decoder.load_state_dict(torch.load('./decoder.pth'))
        self.decoder = self.decoder.cuda()
        self.keyencoder.load_state_dict(torch.load('./key.pth'))
        self.keyencoder = self.keyencoder.cuda()

        if config.attention == 'soft':
            self.AsyAtt = AsyAtt()
        else:
            self.AsyAtt = AsyAttHard()


        S_path = os.path.join(config.style_dir, str(config.S))
        style_images = glob.glob((S_path + '/*.jpg'))
        s = Image.open(style_images[0])
        s = trans(s).cuda()
        self.style_image = s.unsqueeze(0)
        self.style_target = torch.stack([s for i in range(config.batch_size)],0)
    

    def test(self):

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            style_val = self.encoder(self.style_image)
            style_key = self.keyencoder(style_val)

            for filename in self.content_images:
                name = str(filename).split("test_images")[-1][1:].replace("\\", "-")
                name = name.replace("/", "-")

                
                c = Image.open(filename)
                c_tensor = trans(c).unsqueeze(0).cuda()
                val = self.encoder(c_tensor)
                key = self.keyencoder(val)

                content_feature = self.AsyAtt(style_key[0], style_val[0], key, val)
                out = self.decoder(content_feature)

                out = denorm(out).to('cpu')[0]
                c_tensor = denorm(c_tensor).to('cpu')[0]

                if out.shape[1] > c_tensor.shape[1]:
                    c_tensor = torch.cat([c_tensor, torch.zeros([c_tensor.shape[0],out.shape[1]-c_tensor.shape[1],c_tensor.shape[2]])],1)
                elif out.shape[1] < c_tensor.shape[1]:
                    out = torch.cat([out, torch.zeros([out.shape[0],c_tensor.shape[1]-out.shape[1],out.shape[2]])],1)

                save_image(torch.cat([out, c_tensor], 2), os.path.join('./logs/test', name))
            
            


            

                

