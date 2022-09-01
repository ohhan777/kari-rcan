import os
import argparse
import glob
from utils2 import cv2_imwrite
from munch import Munch
import torch
from model.rcan import RCAN
import cv2
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import numpy as np
import torch.nn.functional as F


def tensor_to_cv2img(img):
    if torch.is_tensor(img):
        # denormalization
        # img = img.mul(255).clamp_(0, 255)
        img = img.clamp_(0,255)
        # (C,H,W) to (H,W,C)  
        img = img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)  
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        print('Error! Not a tensor')
        return None


def eval(opt):
    # Network models
    cfg = Munch(n_resgroups=10, n_resblocks=20, n_feats=64, scale=2, rgb_range=255, n_colors=1, reduction=16)
    model = RCAN(cfg)  

    # GPU-support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:   # multi-GPU
        model = torch.nn.DataParallel(model)

    model.to(device)

    if torch.cuda.is_available(): 
            scaler = torch.cuda.amp.GradScaler()
            print('[AMP Enabled]')
    else:
        scaler = None



    # Load weight file
    assert os.path.exists(opt.weights), "no found model weights"
    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    with torch.no_grad():
        # Image preparation
        
        files = glob.glob('data/K3A_png/*.png')
        for file in files:
            lr_img = cv2.imread(file)
            lr_tensor = torch.from_numpy(lr_img[:,:,0].copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            sr_tensor = model(lr_tensor.to(device))
            out_img = sr_tensor[0,0,:,:].clamp(0,255.0).cpu().numpy().astype(np.uint8)
            out_file = file.replace('data/K3A_png', 'out')
            cv2.imwrite(out_file, out_img)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/rcan_best.pth', help='weights file')
    parser.add_argument('--input', default='data/K3A_png/BLD01322_PAN_K3A_NIA0373.png', type=str, help='input image')
    parser.add_argument('--output', default='out.png', type=str, help='input image')
    opt = parser.parse_args()
    eval(opt)


    