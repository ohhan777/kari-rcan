from configparser import Interpolation
import os
import cv2
import torch
import numpy as np
import rasterio
from rasterio.transform import Affine
import skimage

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_images(hr_imgs, sr_imgs, filename):
    # Plot image grid with labels
    size = hr_imgs.size(0)
    for i in range(size):
        if isinstance(hr_imgs, torch.Tensor):
            hr_img = hr_imgs[i].cpu().float().numpy()   # img: (C, H, W)
        if isinstance(sr_imgs, torch.Tensor):
            sr_img = sr_imgs[i].cpu().float().numpy()   # img: (C, H, W)

        hr_img = hr_img * 255.0
        sr_img = sr_img * 255.0

        hr_img = hr_img.astype(np.uint8)
        sr_img = sr_img.astype(np.uint8)
        
        img = np.concatenate((hr_img, sr_img), axis=2)

        cv2.imwrite(str(filename).replace('.png', f'_{i}.png'), img[0])

def save_images(lr_imgs, sr_imgs, save_dir, lr_filenames):
    # Plot image grid with labels
    size = lr_imgs.size(0)
    for i in range(size):
        if isinstance(lr_imgs, torch.Tensor):
            lr_img = lr_imgs[i].cpu().float().numpy().squeeze(0)   # img: (H, W)
        if isinstance(sr_imgs, torch.Tensor):
            sr_img = sr_imgs[i].cpu().float().numpy().squeeze(0)   # img: (H, W)
        lr_filename = lr_filenames[i]

        lr_img = lr_img * 255.0
        sr_img = sr_img * 255.0

        lr_img = lr_img.astype(np.uint8)
        h, w = sr_img.shape
        lr_img = cv2.resize(lr_img, (h, w),  interpolation=cv2.INTER_CUBIC)
        sr_img = sr_img.astype(np.uint8)

        sr_img = skimage.exposure.match_histograms(sr_img, lr_img)
        
        img = np.concatenate((lr_img, sr_img), axis=1)
        filename = os.path.join(save_dir, os.path.basename(lr_filename))
        cv2.imwrite(str(filename).replace('.png', f'_{i}.png'), img)

def save_tiffs(sr_imgs, save_dir, lr_filenames):
    size = sr_imgs.size(0)
    for i in range(size):
        if isinstance(sr_imgs, torch.Tensor):
            sr_img = sr_imgs[i].cpu().float().numpy().squeeze(0)   # img: (H, W)
        lr_filename = lr_filenames[i]
        
        with rasterio.open(lr_filename) as src:
            geo_trans = src.transform
            crs = src.crs
            metadata = src.meta
            
        sr_img = sr_img * 16383.0
        sr_img = sr_img.astype(np.uint16)
        h, w = sr_img.shape
        filename = os.path.join(save_dir, os.path.basename(lr_filename).replace('.tif', '_x2.tif'))
        
        # Create new geotransform with 2x resolution
        sr_geo_trans = Affine(geo_trans.a / 2, geo_trans.b, geo_trans.c,
                             geo_trans.d, geo_trans.e / 2, geo_trans.f)
        
        # Update metadata for the new file
        new_metadata = metadata.copy()
        new_metadata.update({
            'driver': 'GTiff',
            'height': h,
            'width': w,
            'count': 1,
            'dtype': sr_img.dtype,
            'transform': sr_geo_trans,
            'crs': crs
        })
        
        with rasterio.open(filename, 'w', **new_metadata) as dst:
            dst.write(sr_img, 1)

        
        
        
        
