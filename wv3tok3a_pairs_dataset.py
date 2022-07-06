import os
import torch
import glob
from pathlib import Path
import numpy as np
import cv2
from utils import cv2_imwrite, denorm_tensor
from torchvision.utils import save_image
import random


class WV3ToK3APairsDataset(torch.utils.data.Dataset):
    def __init__(self, root, crop_size, upscale_factor, train=False):
        self.root = Path(root)
        self.train = train
        self.crop_size = crop_size - (crop_size % upscale_factor) # changes crop_size to be a multiple of upscale_factor
        self.upscale_factor = upscale_factor
        if train:
            self.img_dir = self.root/Path('wv3_to_k3a_pairs/train/fake_lr')  
        else:
            self.img_dir = self.root/Path('wv3_to_k3a_pairs/val/fake_lr')  
        self.img_files = sorted(glob.glob(os.path.join(str(self.img_dir), '*.png')))
        self.img_size = len(self.img_files)

    def __getitem__(self, idx):
        lr_img_file = self.img_files[idx]
        hr_img_file = lr_img_file.replace('fake_lr', 'hr')
        lr_img = cv2.imread(lr_img_file)
        hr_img = cv2.imread(hr_img_file)
        h, w = lr_img.shape[:2]
        if min(h, w) < self.crop_size:
            print("Error! The image is smaller than the crop size.", lr_img_file)
            return None
        # transforms (crop, resize for lr_img, normalization, and tensor conversion)
        lr_img, hr_img = self.transforms(lr_img[:,:,0], hr_img[:,:,0], is_flip=self.train)  # tensor images in a range of [-1.0, 1.0]
        return lr_img, hr_img
    
    def __len__(self):
        return len(self.img_files)
    
    def transforms(self, lr_img, hr_img, is_flip=False):
        # random crop with respect to high resolution image
        h, w = hr_img.shape[:2] 
        crop_size = self.crop_size
        upscale_factor = self.upscale_factor
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        x = x - (x % upscale_factor) if x % upscale_factor != 0 else x
        y = y - (y % upscale_factor) if y % upscale_factor != 0 else y
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        hr_img = hr_img[y:y+crop_size, x:x+crop_size]
        x = x//upscale_factor
        y = y//upscale_factor
        crop_size = crop_size//upscale_factor
        lr_img = lr_img[y:y+crop_size, x:x+crop_size]
        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            lr_img = lr_img[:,::flip]
            hr_img = hr_img[:,::flip]
        lr_tensor = torch.from_numpy(lr_img.copy()).type(torch.FloatTensor).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_img.copy()).type(torch.FloatTensor).unsqueeze(0)
        return lr_tensor, hr_tensor

   
if __name__ == "__main__":
    dataset = WV3ToK3APairsDataset('./data', 256, 2, train=True)
    lr_img, hr_img = dataset[0]
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    for x, y in loader:
        print(x.shape)
        save_image(x/255.0, "x.png")
        save_image(y/255.0, "y.png")
        import sys
        sys.exit()
    
