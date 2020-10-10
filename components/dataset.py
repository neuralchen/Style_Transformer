import torch
import torch.utils.data as data
import glob
import torchvision.transforms as transforms
from PIL import Image
import random


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

default_trans = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

class HDR_LDR(data.Dataset):
    def __init__(self, path_ldr, path_hdr, trans=default_trans):
        ldrs = glob.glob(path_ldr+'/*.jpg')
        hdrs = glob.glob(path_hdr+'/*.jpg')
        self.dataset = []
        for path in ldrs:
            mirror = path.replace(path_ldr, path_hdr)
            if mirror in hdrs:
                self.dataset.append([path, mirror])
        print(len(self.dataset))
        self.trans = trans


    def __getitem__(self, index):
        ldr_path, hdr_path = self.dataset[index]
        ldr = Image.open(ldr_path)
        hdr = Image.open(hdr_path)
        ref_id = index
        while ref_id == index:
            ref_id = random.randint(1,3000) 
        ref = Image.open(self.dataset[ref_id][1])
        return self.trans(ldr),self.trans(hdr),self.trans(ref)

    def __len__(self):
        return len(self.dataset)

if __name__ == '__main__':
    dataset = HDR_LDR('H:/adobe5k/identity/input', 'H:/adobe5k/identity/output')
    


