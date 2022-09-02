import cv2
import torch
import numpy as np

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