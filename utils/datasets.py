import os
import hashlib
from pathlib import Path
import numpy as np
import random
import glob
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from utils.torch_utils import torch_distributed_zero_first
from utils.augmentations import Albumentations, letterbox, random_perspective, augment_hsv, random_scale, random_crop
import rasterio

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def create_dataloader(path, task, batch_size, hyp=None, augment=False,
                      cache=False, pad=0.0, rank=-1, workers=8,
                      shuffle=True):
    #with torch_distributed_zero_first(rank):
    dataset = LoadImagesAndLabels(path, task, batch_size, augment=augment,
                                      hyp=hyp, cache_imgs=cache)
    batch_size = min(batch_size, len(dataset))
    num_devices = torch.cuda.device_count()
    num_workers = min([os.cpu_count() // max(num_devices, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
                        num_workers=num_workers, sampler=sampler, pin_memory=True, drop_last=False)
    return loader, dataset


class LoadImagesAndLabels(Dataset):
    cache_version = 0.1   # dataset labels *.cache version

    def __init__(self, path, task, batch_size=16, augment=False, 
                 hyp=None, cache_imgs=False):

        self.augment = augment
        self.hyp = hyp
        self.train = True if task == 'train' else False
        self.crop_size = hyp['crop_size'] - ( hyp['crop_size'] % hyp['up_scale'])
        self.upscale_factor = hyp['up_scale']
        self.root = Path(path)
        self.rgb_range = hyp['rgb_range']
        self.task = task
        if task == 'train':
            self.img_dir = self.root/Path('wv3_to_k3a_pairs/train/fake_lr')  
            #self.img_dir = self.root/Path('k3a_to_k3_pairs/train/fake_lr')  
        elif task == 'val':
            self.img_dir = self.root/Path('wv3_to_k3a_pairs/val/fake_lr')
            #self.img_dir = self.root/Path('k3a_to_k3_pairs/val/fake_lr')  
        elif task == 'test':
            self.img_dir = self.root/Path('wv3_to_k3a_pairs/test')  
            #self.img_dir = self.root/Path('k3a_to_k3_pairs/test')

        self.img_files = sorted(glob.glob(os.path.join(str(self.img_dir), '*.tif')))
        self.img_size = len(self.img_files)
                   
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        lr_img_file = self.img_files[idx]
        
        if self.task in ['train', 'val']: 
            lr_img = cv2.imread(lr_img_file, -1).astype(np.float32)
            h, w = lr_img.shape[:2]
            if min(h, w) < self.crop_size:
                print("Error! The image is smaller than the crop size.", lr_img_file)
                return None        
            hr_img_file = lr_img_file.replace('fake_lr', 'hr')
            hr_img = cv2.imread(hr_img_file, -1).astype(np.float32)
            # transforms (crop, resize for lr_img, normalization, and tensor conversion)
            lr_img, hr_img = self.transforms(lr_img, hr_img, is_flip=self.train) 
            return lr_img, hr_img
        if self.task == 'test':
            with rasterio.open(lr_img_file) as src:
                lr_img = src.read(1).astype(np.uint16)/64.0
            lr_tensor = torch.from_numpy(lr_img).type(torch.FloatTensor).unsqueeze(0)
            return lr_tensor, lr_img_file
        

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
        hr_img = hr_img[y:y+crop_size, x:x+crop_size]/64.0
        x = x//upscale_factor
        y = y//upscale_factor
        crop_size = crop_size//upscale_factor
        lr_img = lr_img[y:y+crop_size, x:x+crop_size]/64.0
        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            lr_img = lr_img[:,::flip]
            hr_img = hr_img[:,::flip]
        lr_tensor = torch.from_numpy(lr_img.copy()).type(torch.FloatTensor).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_img.copy()).type(torch.FloatTensor).unsqueeze(0)
        return lr_tensor, hr_tensor



