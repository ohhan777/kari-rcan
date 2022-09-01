import os
import torch
import glob
from pathlib import Path
import numpy as np
import cv2
from utils2 import cv2_imwrite, denorm_tensor
from torchvision.utils import save_image
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, root, crop_size, upscale_factor, train=False):
        self.root = Path(root)
        self.train = train
        self.crop_size = crop_size - (crop_size % upscale_factor) # changes crop_size to be a multiple of upscale_factor
        if train:
            self.img_dir = self.root/Path('DIV2K/DIV2K_train_HR')  
        else:
            self.img_dir = self.root/Path('DIV2K/DIV2K_valid_HR')   
        self.img_files = sorted(glob.glob(os.path.join(str(self.img_dir), '*.png')))
        self.img_size = len(self.img_files)
        self.transforms = get_transforms(self.crop_size, upscale_factor, train)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
        h, w, _ = img.shape
        if min(h, w) < self.crop_size:
            print("Error! The image is smaller than the crop size.", img_file)
            return None
        # transforms (crop, resize for lr_img, normalization, and tensor conversion)
        lr_img, hr_img = self.transforms(img)  # tensor images in a range of [-1.0, 1.0]
        return lr_img, hr_img
    
    def __len__(self):
        return len(self.img_files)

# A.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),

class ImgPreprocess:
    def __init__(self, crop_size, upscale_factor, train):
        rescale_size = crop_size//upscale_factor
        self.downscale = A.Resize(rescale_size, rescale_size, interpolation=cv2.INTER_CUBIC)
        # self.normalize = A.Normalize([0.0, 0.0, 0.0],[1.0, 1.0, 1.0])
        self.to_tensor = ToTensorV2()
        if train:
            self.crop = A.Compose([A.RandomCrop(crop_size, crop_size)],
                                   additional_targets={'image0':'image'})
        else:
            self.crop = A.CenterCrop(crop_size, crop_size)              

    def __call__(self, img):
        cropped_img = self.crop(image=img)['image']
        
        #hr_img = self.normalize(image=cropped_img)['image']   # [-1.0, 1.0]
        hr_img = self.to_tensor(image=cropped_img)['image']          
        
        lr_img = self.downscale(image=cropped_img)['image']    
        # lr_img = self.normalize(image=lr_img)['image']        # [-1.0, 1.0]
        lr_img = self.to_tensor(image=lr_img)['image']
    
        return lr_img.type(torch.FloatTensor), hr_img.type(torch.FloatTensor)  # ByteTensor > FlatTensor


def get_transforms(crop_size, upscale_factor, train=False):
    transforms = ImgPreprocess(crop_size, upscale_factor, train)
    return transforms

   
if __name__ == "__main__":
    dataset = DIV2KDataset('./data', 256, 4, train=True)
    lr_img, hr_img = dataset[0]
    cv2_imwrite('img0.png', lr_img)
    cv2_imwrite('img1.png', hr_img)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    for x, y in loader:
        print(x.shape)
        save_image(x/255.0, "x.png")
        save_image(y/255.0, "y.png")
        import sys
        sys.exit()
    
