import argparse
import time
import warnings
from tqdm import trange
from datetime import datetime
import numpy as np

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils

from network.generator import Generator
from network.discriminator import Discriminator

from data_loader import CelebAHQ
from ops import compute_grad_gp
from utils import *


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("[!] Building model ... ", end=' ', flush=True)

    networks = [
        Discriminator(img_size=64, sn=True),
        Generator(latent_size=args.latent_size)
    ]

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            networks = [x.cuda(args.gpu) for x in networks]
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)

            networks = [torch.nn.parallel.DistributedDataParallel(x, device_ids=[args.gpu]) for x in networks]
        else:
            networks = [x.cuda(args.gpu) for x in networks]
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            networks = [torch.nn.parallel.DistributedDataParallel(x) for x in networks]

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        networks = [x.cuda(args.gpu) for x in networks]
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        networks = [x.cuda() for x in networks]

    discriminator, generator = networks

    print("Done !")

    print("[!] Building optimizer ... ", end=' ', flush=True)
    criterion = nn.BCELoss().cuda(args.gpu)

    d_opt = torch.optim.Adam(discriminator.parameters(), args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    g_opt = torch.optim.Adam(generator.parameters(), args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

    print("Done !")

    if args.load_model is not None:
        print("[!] Restoring model ... ")

        check_load = open(os.path.join(args.log_dir, "checkpoint.txt"), 'r')
        to_restore = check_load.readlines()[-1].strip()
        load_file = os.path.join(args.log_dir, to_restore)
        if os.path.isfile(load_file):
            print(" => loading checkpoint '{}'".format(load_file))
            checkpoint = torch.load(load_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            discriminator.load_state_dict(checkpoint['D_state_dict'])
            generator.load_state_dict(checkpoint['G_state_dict'])
            d_opt.load_state_dict(checkpoint['d_optimizer'])
            g_opt.load_state_dict(checkpoint['g_optimizer'])
            print(" => loaded checkpoint '{}' (epoch {})"
                  .format(load_file, checkpoint['epoch']))
        else:
            print(" => no checkpoint found at '{}'".format(args.log_dir))

    cudnn.benchmark = True

    print("[!] Loading datasets ... ")
    transform = transforms.Compose([transforms.CenterCrop(128),
                                    transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CelebAHQ(transform, 'train', args.batch_size, args.img_dir)
    val_dataset = CelebAHQ(transform, 'val', args.batch_size, args.img_dir)
    print("Done !")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.validation:
        validate(val_loader, discriminator, generator, criterion, 0, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        validate(val_loader, discriminator, generator, criterion, epoch, args)
        train(train_loader, discriminator, generator, criterion, d_opt, g_opt, epoch, args)

        # Save and Validate model only with main gpu
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            with open(os.path.join(args.log_dir, "checkpoint.txt"), "a+") as check_list:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'D_state_dict': discriminator.state_dict(),
                    'G_state_dict': generator.state_dict(),
                    'd_optimizer': d_opt.state_dict(),
                    'g_optimizer': g_opt.state_dict(),
                }, check_list, args.log_dir, epoch + 1)
            print('')


def train(train_loader, D, G, criterion, d_opt, g_opt, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    d_losses = AverageMeter()
    d_regs = AverageMeter()
    g_losses = AverageMeter()

    # switch to train mode

    D.train()
    G.train()

    end = time.time()
    train_it = iter(train_loader)
    t_train = trange(0, len(train_loader), initial=0, total=len(train_loader))
    # for i, (x_real, _) in enumerate(train_loader):
    for i in t_train:
        x_real = next(train_it)
        # measure data loading time
        data_time.update(time.time() - end)

        d_opt.zero_grad()

        z_input = torch.randn(args.batch_size, args.latent_size)
        ones = torch.ones(args.batch_size, 1)
        zeros = torch.zeros(args.batch_size, 1)

        if args.gpu is not None:
            x_real = x_real.cuda(args.gpu, non_blocking=True)
            z_input = z_input.cuda(args.gpu, non_blocking=True)
            ones = ones.cuda(args.gpu, non_blocking=True)
            zeros = zeros.cuda(args.gpu, non_blocking=True)

        x_real.requires_grad_()

        d_real_logit = D(x_real)
        d_adv_real = criterion(d_real_logit, ones)
        d_adv_real.backward(retain_graph=True)

        # D regularization, R1 - zero centered GP with real data
        reg = 5.0 * compute_grad_gp(d_real_logit, x_real).mean()
        reg.backward()

        # Train D with FAKE
        x_fake = G(z_input)
        d_fake_logit = D(x_fake.detach())
        d_adv_fake = criterion(d_fake_logit, zeros)
        d_adv_fake.backward()

        d_loss = d_adv_real + d_adv_fake
        d_opt.step()

        g_opt.zero_grad()
        x_fake = G(z_input)
        g_fake_logit = D(x_fake)
        g_adv = criterion(g_fake_logit, ones)
        g_loss = g_adv
        g_loss.backward()
        g_opt.step()

        # measure accuracy and record loss
        d_losses.update(d_loss.item(), x_real.size(0))
        d_regs.update(reg.item(), x_real.size(0))
        g_losses.update(g_loss.item(), x_real.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_step == 0:
            t_train.set_description('Epoch: [{}/{}], '
                                    'Loss: D[{d_losses.avg:.3f}] '
                                    'G[{g_losses.avg:.3f}]'.format(epoch, args.epochs,
                                                                   d_losses=d_losses, g_losses=g_losses))


def validate(data_loader, D, G, criterion, epoch, args):
    # switch to evaluate mode
    D.eval()
    G.eval()

    with torch.no_grad():
        z_val = torch.randn(args.batch_size, args.latent_size)
        if args.gpu is not None:
            z_val = z_val.cuda(args.gpu, non_blocking=True)

        x_fake_val = G(z_val)

        vutils.save_image(x_fake_val, os.path.join(args.res_dir, '{}_{}_fake.png'.format(epoch, args.gpu)),
                          normalize=True, nrow=int(np.sqrt(args.batch_size)))
