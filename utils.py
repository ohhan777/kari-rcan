import torch
import numpy as np
import cv2
from torchvision.utils import save_image
import albumentations as A
import torch.nn.functional as F

def cv2_imwrite(filename, img):
   img = tensor_to_cv2img(img)
   cv2.imwrite(filename, img)

# [-1.0, 1.0] to [0.0, 1.0]
def denorm_tensor(img):
    return img #img* 0.5 + 0.5 

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
    


def bgr2y(bgr_img):
    bgr_img = bgr_img/255.0
    y_img = bgr_img[:, :, 2] * 65.481 + bgr_img[:, :, 1] * 128.553 + bgr_img[:, :, 0] * 24.966 + 16
    return y_img
   
@torch.no_grad()
def get_tensor_psnr(img1, img2, boader_crop=0, y_channel_only=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Args:
        img1 (tensor): Images with range [0.0, 1.0]. (N,C,H,W) or (C, H, W) in RGB order
        img2 (tensor): Images with range [0.0, 1.0]. (N,C,H,W) or (C, H, W) in RGB order
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        y_channel_only (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """
    assert(img1.max() <= 1.0 and img1.min() >= 0.0), 'range is expected to be [0.0, 1.0]'  
    if boader_crop != 0:
        img1 = img1[...,boader_crop:-boader_crop,boader_crop:-boader_crop]
        img2 = img2[...,boader_crop:-boader_crop,boader_crop:-boader_crop]

    if y_channel_only:
        if img1.shape[1] != 1:
            img1 = (img1[...,0,:,:] * 65.481 + img1[...,1,:,:] * 128.553 + img1[...,2,:,:] * 24.966 + 16)/255.0
            img2 = (img2[...,0,:,:] * 65.481 + img2[...,1,:,:] * 128.553 + img2[...,2,:,:] * 24.966 + 16)/255.0
        
    return 10* torch.log10(1.0/((img1-img2)**2).mean()).item()


@torch.no_grad()
def get_tensor_ssim(img1, img2, boader_crop=0, y_channel_only=False):
    """Calculate SSIM (structural similarity).
    Args:
        img1 (tensor): Images with range [0.0, 1.0]. (N,C,H,W) or (C, H, W) in RGB order
        img2 (tensor): Images with range [0.0, 1.0]. (N,C,H,W) or (C, H, W) in RGB order
    Returns:
        float: ssim result.
    Remarks:
        This code is borrowed from pytorch_ssim git repository.
    """
    assert(img1.max() <= 1.0 and img1.min() >= 0.0), 'range is expected to be [0.0, 1.0]'  
    if boader_crop != 0:
        img1 = img1[...,boader_crop:-boader_crop,boader_crop:-boader_crop]
        img2 = img2[...,boader_crop:-boader_crop,boader_crop:-boader_crop]
        
    if y_channel_only:
        if img1.shape[1] != 1:
            img1 = (img1[...,0,:,:] * 65.481 + img1[...,1,:,:] * 128.553 + img1[...,2,:,:] * 24.966 + 16)/255.0
            img2 = (img2[...,0,:,:] * 65.481 + img2[...,1,:,:] * 128.553 + img2[...,2,:,:] * 24.966 + 16)/255.0
    
    for _ in range(4 - img1.dim()):
        img1 = torch.unsqueeze(img1, 0)
        img2 = torch.unsqueeze(img2, 0)

    channel = img1.size(1)
    window_size = 11
    window = _create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        window = window.type_as(img1)

    padd = 0

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity
    cs = cs.mean()
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    return ssim_map.mean().item()





def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


    



def save_val_images(val_dataloader, G_AB, G_BA, epoch, device):
    a, b = next(iter(val_dataloader))
    a, b = a.to(device), b.to(device)
    G_AB.eval()
    G_BA.eval()
    with torch.no_grad():
        fake_b = G_AB(a)
        fake_a = G_BA(b)
        save_image(b * 0.5 + 0.5, f"orig_b_epoch_{epoch}.png")
        save_image(a * 0.5 + 0.5, f"orig_a_epoch_{epoch}.png")
        save_image(fake_b * 0.5 + 0.5, f"fake_b_epoch_{epoch}.png")
        save_image(fake_a * 0.5 + 0.5, f"fake_a_epoch_{epoch}.png")
        


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg