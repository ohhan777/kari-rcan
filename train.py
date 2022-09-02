import os
import math
import yaml
import argparse
import torch
import torchvision
from model.rcan import RCAN
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import time
from pathlib import Path
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from utils.loggers import Loggers
#from utils import AverageMeter, denorm_tensor, get_tensor_ssim, get_tensor_psnr
from utils.datasets import create_dataloader
from utils.general import (print_args, LOGGER, colorstr, one_cycle, increment_path,
                           check_yaml, methods, check_suffix, init_seeds, intersect_dicts,
                           strip_optimizer, get_latest_run, FullModel, AverageMeter, check_version)
from utils.torch_utils import select_device, de_parallel, is_main_process, EarlyStopping, is_distributed, get_rank, get_world_size, reduce_tensor
from utils.callbacks import Callbacks
from utils.metrics import fitness
import numpy as np

from copy import deepcopy
from datetime import datetime
import val


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

torch.manual_seed(0)


def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, resume, weights, noval, nosave, workers = Path(opt.save_dir), opt.epochs, opt.batch_size,\
                                                                opt.resume, opt.weights, opt.noval, opt.nosave, opt.workers

    # Directories
    w = save_dir / 'weights'
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    if is_main_process():
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir/ 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    
    # Loggers
    data_dict = None
    if is_main_process():
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    cuda = device.type != 'cpu'    
    init_seeds(1 + RANK)
    # TODO: data_dict

    # Model
    check_suffix(weights, '.pt')  # check weights
    model = RCAN(hyp)

    # Dataloaders and Datasets
    train_loader, train_dataset = create_dataloader('./data', is_train=True, batch_size=batch_size // WORLD_SIZE,
                                                    hyp=hyp, augment=True, cache=False, rank=LOCAL_RANK, workers=workers)
    
    val_loader, _ = create_dataloader('./data', is_train=False, batch_size=batch_size // WORLD_SIZE,
                                                    hyp=hyp, augment=False, cache=False, rank=LOCAL_RANK, workers=workers)
    nb = len(train_loader)  # number of batches
    loss_func = nn.L1Loss()
    model = model.to(device)

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    if is_main_process():
        LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    

    if opt.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    
    # # Scheduler
    # if opt.linear_lr:
    #     lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # # if opt.cos_lr:
    # #     lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    # else:
    #     lf = lambda x: math.pow(1 -x / epochs, hyp['poly_exp'])    
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  

    # Resume (load checkpoint and apply it to the model)
    start_epoch, best_fitness, best_psnr, best_epoch = 0, 0.0, 0.0, 0
    if resume:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict())  # intersect
        model.load_state_dict(csd, strict=False)  # load

        if is_main_process():
            LOGGER.info(colorstr('yellow', ('Resuming training from %s saved at %s (last epoch %d)') 
                                            % (weights, ckpt['date'], ckpt['epoch'])))
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report  

        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            #best_psnr = ckpt['best_psnr']
            best_psnr = best_fitness
            best_epoch = ckpt['best_epoch']
        
        start_epoch = ckpt['epoch'] + 1
        assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'        
        del ckpt, csd
    
     # DP (Data-Parallel) mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        if is_main_process():
            LOGGER.info("DP mode is enabled, but DDP is preferred for best performance." )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        if is_main_process():
            LOGGER.info('Using SyncBatchNorm()')
    
    # DDP mode
    if cuda and RANK != -1:
        if check_version(torch.__version__, '1.11.0'):
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
        else:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


    # Start training 
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    results = [0, 0, 0]   # mIOU, pix_acc, val loss
    #scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    callbacks.run('on_train_start')
    if is_main_process():
        LOGGER.info(f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n' \
                    f"Logging results to {colorstr('bold', save_dir)}\n" \
                    f"Starting training for {epochs} epochs...")
        
    for epoch in range(start_epoch, epochs):  # epoch ---------------------------------------------------------------
        t1 = time.time()
        callbacks.run('on_train_epoch_start', epoch=epoch)
        model.train()
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        avg_loss = AverageMeter()

        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            # Warmup
            # if ni <= nw:
            #     xi = [0, nw]  # x interp
            #     # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            #     accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
            #     for j, x in enumerate(optimizer.param_groups):
            #         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #         x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            #         if 'momentum' in x:
            #             x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled=cuda):
                sr_imgs = model(lr_imgs)            
                loss = loss_func(sr_imgs, hr_imgs)    

                if RANK != -1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    #loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                last_opt_step = ni

            # Log
            if is_main_process() and i % 10 == 0:
                avg_loss.update(loss.item())    # update average loss
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], GPU: {}, ' \
                      'lr: {}, Loss: {:.6f}' .format(
                      epoch, epochs, i, nb, mem, [x['lr'] for x in optimizer.param_groups], avg_loss.average())
                LOGGER.info(msg)
            
            # debug mode 
            if opt.debug and i == 10:  
               break

            # end batch --------------------------------------------------------------------------------------
        
        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        #scheduler.step()
            

        #if is_main_process():
        #    callbacks.run('on_train_epoch_end', epoch=epoch)
        
        # validation
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        if not noval or final_epoch:  # Calculate mIOU, pix_acc, val loss
            results = val.run(model=model, batch_size=batch_size, workers=workers, hyp=hyp, dataloader=val_loader, 
                                  save_dir=save_dir, callbacks=callbacks, loss_func=loss_func)

        if is_main_process():
            fi = fitness(np.array(results[:2]))
            if fi > best_fitness:
                best_fitness = fi 
                best_epoch = epoch    
                best_psnr = results[0]
                LOGGER.info(colorstr('yellow','bold','[Best so far]'))

            log_vals = [loss.item()] + list(results[:3]) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)
            LOGGER.info('Best PSNR=%.4f (epoch=%d)' % (best_psnr, best_epoch))

            # Save model
            if (not nosave) or final_epoch:  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'best_psnr' : best_psnr,
                        'best_epoch': best_epoch, 
                        'model': deepcopy(de_parallel(model)).half(), 
                        'optimizer': optimizer.state_dict(), 
                        'wandb_id' : loggers.wandb.wandb_run.id if loggers.wandb else None,                   
                        'date': datetime.now().isoformat()}
                
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)
            LOGGER.info(f'Epoch {epoch} completed in {(time.time() - t1):.3f} seconds.')

            # Stop Single-GPU
            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if is_main_process():
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
            
    torch.cuda.empty_cache()
    # return results
    return 0
            

    


def main(opt, callbacks=Callbacks()):
    if is_main_process():
        print_args(FILE.stem, opt)
    # Resume
    if opt.resume: # resume an interrupted run
        epochs = opt.epochs
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.weights, opt.resume, opt.epochs = ckpt, True, epochs  # reinstate
    else:
        opt.hyp, opt.weights, opt.project = \
            check_yaml(opt.hyp), str(opt.weights), str(opt.project)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if RANK != -1:
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(RANK)
        device = torch.device('cuda', RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    
    # Train
    train(opt.hyp, opt, device, callbacks)
    
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights.pt', help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT/'runs', help='save to project/runs/name')
    parser.add_argument('--name', default='exp', help='save to project/runs/name')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.rcan.yaml', help='hyperparameters path')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--epochs', type=int, default=700, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--debug', action='store_true', help='debug mode (training is early stopped every epoch)')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
