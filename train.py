import os
import argparse
import torch
import torchvision
from model.rcan import RCAN
from wv3tok3a_pairs_dataset import WV3ToK3APairsDataset
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
import wandb
from utils import AverageMeter, denorm_tensor, get_tensor_ssim, get_tensor_psnr
import tqdm
import numpy as np
from metrics import calculate_psnr, calculate_ssim


torch.manual_seed(0)


def train(opt):

    opt.amp = True
    lr = opt.lr
    # loading a weight file (if exists)
    os.makedirs('weights', exist_ok=True)
    weight_file = Path('weights')/(opt.name + '_latest.pth')
    best_psnr = 0.0
    resume = False
    start_epoch, end_epoch = (0, opt.epochs)
    if opt.resume and os.path.exists(weight_file):
        checkpoint = torch.load(weight_file)
        start_epoch = checkpoint['epoch'] + 1
        resume = True
        lr = opt.lr
        opt = checkpoint['opt']
        best_psnr = checkpoint['best_psnr']
        print('resumed from epoch %d' % start_epoch)

    # wandb settings
    wandb.login()
    wandb.init(project='kari-rcan-k3a-wv3')

    # dataset
    train_dataset = WV3ToK3APairsDataset('./data', opt.crop_size, opt.scale, train=True)
    val_dataset = WV3ToK3APairsDataset('./data', opt.crop_size, upscale_factor=opt.scale, train=False)

    # dataloader
    num_workers = min([min([os.cpu_count(), 8]), opt.batch_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                            shuffle=True, pin_memory=True, num_workers=num_workers)
   
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                            shuffle=False, pin_memory=True, num_workers=num_workers)

    # Network models
    model = RCAN(opt)  

    # GPU-support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:   # multi-GPU
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    wandb.watch(model)

    # loss function
    l1_loss_fn = nn.L1Loss()

    # learning rate


    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    if torch.cuda.is_available() and opt.amp == True:
        scaler = torch.cuda.amp.GradScaler()
        print('[AMP Enabled]')
    else:
        scaler = None

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5e3, gamma=0.75)

    # loading a weight file (if exists)
    if resume and os.path.exists(weight_file):
        model.load_state_dict(checkpoint['model'])
        print('model loaded')

    
    # training
    for epoch in range(start_epoch, end_epoch):
        print('epoch: %d/%d, learning rate:%.8f' % (epoch+1, end_epoch, lr))
        t0 = time.time()
        train_one_epoch(train_dataloader, model, optimizer, l1_loss_fn, lr_scheduler, device, scaler)
        t1 = time.time()
        train_time = t1 - t0
        # save the current weights
        lr = lr_scheduler.get_last_lr()[0]
        state = {'model': model.state_dict(), 'epoch': epoch, 'opt': opt, 'lr': lr, 'best_psnr': best_psnr}
        torch.save(state, weight_file)
        t0 = time.time()
        print('[validation for sample images]')
        if epoch % 20 == 0:
            psnr_val, ssim_val = val_one_epoch(val_dataloader, epoch, model, device, save_imgs=True, sample=True)
        else:
            psnr_val, ssim_val = val_one_epoch(val_dataloader, epoch, model, device, save_imgs=False, sample=True)
        wandb.log({'psnr_val': psnr_val, 'ssim_val': ssim_val})
        t1 = time.time()
        val_time = t1 - t0
        if psnr_val > best_psnr:
            best_weight_file = Path('weights')/(opt.name + '_best.pth')
            state = {'model': model.state_dict(), 'epoch': epoch, 'opt': opt, 'lr': lr, 'best_psnr': best_psnr}
            torch.save(state, best_weight_file)
            print('best PSNR=>saved')
       
            
        print('train time=%.2f, val time=%.2f' % (train_time, val_time))

        

def train_one_epoch(dataloader, model, optimizer, l1_loss_fn, lr_scheduler, device, scaler=None):
    model.train()

    for i, (lr_imgs, hr_imgs) in enumerate(dataloader):

        hr_imgs = hr_imgs.to(device)    
        lr_imgs = lr_imgs.to(device)
        optimizer.zero_grad()
        avg_loss = AverageMeter()
        if scaler is None:
            sr_imgs = model(lr_imgs)            
            loss = l1_loss_fn(sr_imgs, hr_imgs)
              
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                sr_imgs = model(lr_imgs)            
                loss = l1_loss_fn(sr_imgs, hr_imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        avg_loss.update(loss.item())  
        lr_scheduler.step()

        with torch.no_grad():
            if i % 5 == 0:
                print('[%d/%d] loss: %.6f' % 
                       (i+1, len(dataloader), avg_loss.value()))
                wandb.log({"loss": avg_loss.value()})
        
    
def val_one_epoch(dataloader, epoch, model, device, save_imgs=True, sample=False):
    model.eval()
    psnr_sr_list = []
    psnr_blr_list = []
    ssim_list = []
    val_imgs = []
    diff_list = []
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            blr_imgs = torch.nn.functional.interpolate(lr_imgs, size=hr_imgs.size(2),  mode='bicubic', align_corners=True)
            
            blr_imgs = (blr_imgs.clamp_(0.0,255.0)/255.0).to(device)
            hr_imgs = hr_imgs/255.0
            sr_imgs = sr_imgs.clamp_(0.0,255.0)/255.0

            psnr_sr = get_tensor_psnr(sr_imgs, hr_imgs, y_channel_only=True)
            psnr_blr = get_tensor_psnr(blr_imgs, hr_imgs, y_channel_only=True)
            ssim = get_tensor_ssim(sr_imgs, hr_imgs, y_channel_only=True)
            psnr_sr_list.append(psnr_sr)
            psnr_blr_list.append(psnr_blr)
            ssim_list.append(ssim)

            
            if save_imgs:
                val_imgs.extend([blr_imgs.squeeze(0).cpu(), hr_imgs.squeeze(0).cpu(), sr_imgs.squeeze(0).cpu()])
        epoch_psnr_sr = np.mean(psnr_sr_list)
        epoch_psnr_blr = np.mean(psnr_blr_list)
        epoch_ssim = np.mean(ssim_list)
        print("[PSNR=%.4f dB(BLR %.4f dB), SSIM=%.4f]" % (epoch_psnr_sr, epoch_psnr_blr, epoch_ssim))
        if sample:
            wandb.log({"PSNR(Y)": epoch_psnr_sr, "PSNR_Bicubic(Y)": epoch_psnr_blr, "SSIM(Y)": epoch_ssim})
        if save_imgs:
            val_imgs = torch.stack(val_imgs)
            num_chunks = -(-val_imgs.size(0)//6)   # equivalent to math.ceil(val_imgs.size(0)/8)
            val_imgs = torch.chunk(val_imgs, num_chunks)
            for i, val_img in enumerate(val_imgs):
                grid_img = torchvision.utils.make_grid(val_img, nrow=3, padding=5)
                torchvision.utils.save_image(grid_img, 'val_imgs_epoch_%d_index_%d.png' % (epoch+1, i))
    return epoch_psnr_sr, epoch_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int,
                        default=700, help='target epochs')
    parser.add_argument('--batch_size', type=int,
                        default=12, help='batch size')
    parser.add_argument('--name', default='rcan', help='name for the run')
    parser.add_argument('--crop_size', default=512, type=int, help='training images crop size')
    parser.add_argument('--scale', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
    parser.add_argument('--n_resgroups', default=10, type=int, help='number of residual groups')
    parser.add_argument('--n_resblocks', default=20, type=int, help='number of residual blocks')
    parser.add_argument('--n_feats', default=64, type=int, help='number of features')
    parser.add_argument('--rgb_range', default=255, type=int, help='number of residual blocks')
    parser.add_argument('--n_colors', default=1, type=int, help='number of color channels')
    parser.add_argument('--reduction', default=16, type=int, help='reduction factor')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    
    parser.add_argument('--resume', action='store_true', help='resuming from checkpoint file')
    parser.add_argument('--amp', action='store_true', help='AMP for speed-up')

    opt = parser.parse_args()

    train(opt)
