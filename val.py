import argparse
import json
import os
import yaml
import sys
from model.rcan import RCAN
from pathlib import Path
from threading import Thread
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from metrics import get_tensor_psnr, get_tensor_ssim
from utils.datasets import create_dataloader
from utils.callbacks import Callbacks
from utils.plots import plot_images
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.general import LOGGER, AverageMeter, colorstr, print_args, check_version, increment_path, intersect_dicts
from utils.plots import plot_images, save_images, save_tiffs
from utils.torch_utils import is_main_process, select_device, time_sync, reduce_tensor
from torchvision import models
import skimage

# DDP-related settings
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

@torch.no_grad()
def run(weights=None,
        batch_size=32,
        task='val',
        device='',
        workers=8,
        hyp=None,
        half=False,  # use FP16 half-precision inference
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        model=None,
        dataloader=None,
        save_dir=Path(''),
        callbacks=Callbacks(),
        loss_func=None, 
        ):

    # DDP-related settings
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
    RANK = int(os.getenv('RANK', -1))
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

    # Initialize/load model and set device
    training = model is not None
    if training:
        device = next(model.parameters()).device      
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()

    else: # called directly
        cuda = device.type != 'cpu'
        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)

               
        # Load model
        model = RCAN(hyp)
        
        test_loader, _ = create_dataloader('./data', task='test', batch_size=batch_size // WORLD_SIZE,
                                                    hyp=hyp, augment=False, cache=False, rank=LOCAL_RANK, workers=workers)
        nb = len(test_loader)  # number of batches
        model = model.to(device)

        #half &= device.type != 'cpu'  # half precision only supported on CUDA
        #model.half() if half else model.float()

        # Load checkpoint and apply it to the model
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict())  # intersect
        model.load_state_dict(csd, strict=False)  # load

        if is_main_process():
            LOGGER.info(colorstr('yellow', ('Resuming training from %s saved at %s (last epoch %d)') 
                                            % (weights, ckpt['date'], ckpt['epoch'])))
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report  
        
        del ckpt, csd

    # Configure
    model.eval()
   
    
    seen = 0  # number of images seen
    dt = [0.0, 0.0, 0.0]
    
    callbacks.run('on_val_start')
    np.seterr(invalid='ignore')
    

    if task in ('train', 'val'):
        avg_loss = AverageMeter()
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()

        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            nb = len(dataloader)   # number of batches
            callbacks.run('on_val_batch_start')
            seen += lr_imgs.shape[0]
            size = lr_imgs.size()
            t1 = time_sync()
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            lr_imgs = lr_imgs.half() if half else lr_imgs.float()    # half precision (fp16)
            hr_imgs = hr_imgs.half() if half else hr_imgs.float()    # half precision (fp16)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            sr_imgs = model(lr_imgs)
            loss = loss_func(sr_imgs, hr_imgs)  
            dt[1] += time_sync() - t2

            if training and RANK != -1:
                dist.all_reduce(loss)   # in DDP mode, losses in all processes are summed

            avg_loss.update(loss.item())

            # metric calculation
            sr_imgs.clamp_(0.0, 255.0).div_(255.0)
            hr_imgs.div_(255.0)
            

            # PSNR
            psnr = get_tensor_psnr(sr_imgs, hr_imgs, y_channel_only=True)
            avg_psnr.update(psnr)

            # SSIM
            ssim = get_tensor_ssim(sr_imgs, hr_imgs, y_channel_only=True)
            avg_ssim.update(ssim)

            t3 = time_sync()
            dt[2] += time_sync() - t3

            if is_main_process():
                if i == 0:
                    LOGGER.info(colorstr('validation: '))
                if i % 5 == 0:
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)               
                    LOGGER.info('            [%d/%d] GPU: %s, loss: %.4f' % (i, nb, mem, avg_loss.average())) 
                if i < 5:
                    filename = save_dir / f'val_batch{i}.png'
                    plot_images(hr_imgs, sr_imgs, filename)

            #callbacks.run('on_val_batch_end')
        
        # Metrics
        if training:
            if (RANK != -1):   # in DDP mode, psnrs and ssim in all processes are averaged
                total_avg_loss = torch.tensor(avg_loss.average()).to(device)
                dist.all_reduce(total_avg_loss, op=dist.ReduceOp.SUM)
                total_avg_loss /= float(dist.get_world_size())
                total_avg_loss = total_avg_loss.cpu().item()

                total_avg_psnr = torch.tensor(avg_psnr.average()).to(device)
                dist.all_reduce(total_avg_psnr, op=dist.ReduceOp.SUM)
                total_avg_psnr /= float(dist.get_world_size())
                total_avg_psnr = total_avg_psnr.cpu().item()

                total_avg_ssim = torch.tensor(avg_ssim.average()).to(device)
                dist.all_reduce(total_avg_ssim, op=dist.ReduceOp.SUM)
                total_avg_ssim /= float(dist.get_world_size())
                total_avg_ssim = total_avg_ssim.cpu().item()

            elif (RANK == -1):
                total_avg_loss = avg_loss.average()
                total_avg_psnr = avg_psnr.average()
                total_avg_ssim = avg_ssim.average()

        
        #callbacks.run('on_val_end')

        model.float()  # return to float32 for training

        # Print results
        if is_main_process():
            LOGGER.info(f'PSNR: %.4f, SSIM: %4f, Loss: %.4f' % (total_avg_psnr, total_avg_ssim, total_avg_loss))
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms confusion matrix calculation per image' % t)

        return (total_avg_psnr, total_avg_ssim, total_avg_loss)
    
    elif task == 'test':
        for i, (lr_imgs, lr_filenames) in enumerate(test_loader):
            callbacks.run('on_val_batch_start')
            seen += lr_imgs.shape[0]
            size = lr_imgs.size()
            t1 = time_sync()
            lr_imgs = lr_imgs.to(device)
            #lr_imgs = lr_imgs.half() if half else lr_imgs.float()    # half precision (fp16)
            
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            sr_imgs = model(lr_imgs)
            dt[1] += time_sync() - t2
            # metric calculation
            lr_imgs.div_(255.0)
            sr_imgs.div_(255.0).clamp_(0.0, 1.0)

            
            t3 = time_sync()
            dt[2] += time_sync() - t3

            save_images(lr_imgs, sr_imgs, save_dir, lr_filenames)
            #save_tiffs(sr_imgs, save_dir, lr_filenames)

        return (0, 0, 0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if RANK != -1:
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(RANK)
        device = torch.device('cuda', RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    opt.task = 'test'
    root_path = os.path.dirname(str(Path(opt.weights).parent))
    opt.hyp = os.path.join(root_path, 'hyp.yaml')
    opt.save_dir = os.path.join(root_path, 'results')
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    opt.device = device

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    else:
        # TODO: python val.py --task speed --weights weights0.pt weights1.pt ...
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        for opt.weights in weights:
            run(**vars(opt), plots=False)
    
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)