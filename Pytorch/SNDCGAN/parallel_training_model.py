import os
import numpy as np
import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from network.discriminator import Discriminator
from network.generator import Generator
from utils import AverageMeter, map_dict, save_checkpoint
from ops import compute_zero_gp, accumulate
from data_loader import CelebAANNO

from metrics.fid_score import FID


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    cudnn.benchmark = True

    if args.gpu is not None:
        print("Using GPU:{} for training.".format(args.gpu))

    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    is_main = not args.distributed or args.rank % ngpus_per_node == 0

    if is_main: print("[!] Building model ... ", end=' ', flush=True)
    networks = dict(
        D=Discriminator(img_size=64, sn=True),
        G=Generator(latent_size=args.latent_size)
    )
    if is_main:
        networks_on_main = dict(
            G_running=Generator(latent_size=args.latent_size).train(False)
        )

    if args.distributed:
        args.batch_size = args.gpu_batch_size
        args.workers = args.workers // ngpus_per_node

        torch.cuda.set_device(args.gpu)
        networks = map_dict(
            lambda x: torch.nn.parallel.DistributedDataParallel(x.cuda(args.gpu), device_ids=[args.gpu]),
            networks
        )

        if is_main:
            networks_on_main = map_dict(lambda x: x.cuda(args.gpu), networks_on_main)
            networks.update(networks_on_main)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        networks = map_dict(lambda x: x.cuda(args.gpu), networks)
        if is_main:
            networks_on_main = map_dict(lambda x: x.cuda(args.gpu), networks_on_main)
            networks.update(networks_on_main)

    else:
        networks = map_dict(lambda x: x.cuda(), networks)
        if is_main:
            networks.update(networks_on_main)

    if is_main: accumulate(networks['G_running'], networks['G'].module, 0)

    if is_main: print("Done !")

    if is_main: print("[!] Building optimizer ... ", end=' ', flush=True)
    d_opt = torch.optim.Adam(networks['D'].parameters(), args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    g_opt = torch.optim.Adam(networks['G'].parameters(), args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    optimizers = dict(D=d_opt, G=g_opt)
    if is_main: print("Done !")

    if args.load_model is not None:
        if is_main: print("[!] Restoring model ... ")
        with open(os.path.join(args.load_model, "ckpt", "checkpoint.txt"), "r") as f:
            to_restore = f.readlines()[-1].strip()
            load_file = os.path.join(args.load_model, "ckpt", to_restore)

            if os.path.isfile(load_file):
                if is_main: print(" => loading checkpoint '{}'".format(load_file))
                checkpoint = torch.load(load_file, map_location='cpu')
                args.start_epoch = checkpoint['epoch']
                networks['D'].load_state_dict(checkpoint['discriminator_state_dict'])
                networks['G'].load_state_dict(checkpoint['generator_state_dict'])
                networks['G_running'].load_state_dict(checkpoint['generator_running_state_dict'])
                optimizers['G'].load_state_dict(checkpoint['d_opt'])
                optimizers['D'].load_state_dict(checkpoint['g_opt'])
                if is_main: print(" => loaded checkpoint '{}' (epoch {})".format(load_file, checkpoint['epoch']))

            else:
                if is_main: print(" => no checkpoint found at '{}'".format(args.load_model))

    if is_main: print("[!] Loading dataset ... ")
    transform = transforms.Compose([transforms.CenterCrop(128),
                                    transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CelebAANNO(transform, 'train', args.dataset_dir)
    val_dataset = CelebAANNO(transform, 'val', args.dataset_dir)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             num_workers=args.workers, pin_memory=True, drop_last=True)

    if is_main: print("Done !")

    if args.validation:
        validate(val_loader, networks, 0, args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch % args.val_epoch == 0 and is_main:
            validate(val_loader, networks, epoch, args)

        # train for one epoch
        train(train_loader, networks, optimizers, epoch, args, is_main)

        # Save and validate model only with main gpu
        if is_main:
            with open(os.path.join(args.ckpt_dir, "checkpoint.txt"), "a+") as check_list:
                save_checkpoint(
                    dict(epoch=epoch+1,
                         generator_state_dict=networks['G'].state_dict,
                         generator_running_state_dict=networks['G_running'].state_dict,
                         discriminator_state_dict=networks['D'].state_dict,
                         g_opt=optimizers['G'],
                         d_opt=optimizers['D'],
                         ),
                    check_list,
                    args.ckpt_dir,
                    epoch+1
                )


def train(train_loader, networks, optimizers, epoch, args, is_main=False):
    am_loss_g = AverageMeter()
    am_loss_d = AverageMeter()

    am_mean_r = AverageMeter()
    am_mean_f = AverageMeter()

    networks['G'].train()
    networks['D'].train()

    ones = torch.ones(args.batch_size, 1)
    zeros = torch.zeros(args.batch_size, 1)

    if args.gpu is not None:
        ones = ones.cuda(args.gpu, non_blocking=True)
        zeros = zeros.cuda(args.gpu, non_blocking=True)

    else:
        ones = ones.cuda()
        zeros = zeros.cuda()

    print("", end="", flush=True)
    train_it = iter(train_loader)
    t_train = tqdm.trange(0, args.steps, disable=not is_main)
    for t in t_train:
        am_loss_g.reset()
        am_loss_d.reset()
        am_mean_r.reset()
        am_mean_f.reset()
        for i in range(args.ttur_d):
            try:
                x_real = next(train_it)
            except StopIteration:
                train_it = iter(train_loader)
                x_real = next(train_it)

            z_input = torch.randn(args.batch_size, args.latent_size)
            if args.gpu is not None:
                x_real = x_real.cuda(args.gpu, non_blocking=True)
                z_input = z_input.cuda(args.gpu, non_blocking=True)
            else:
                x_real = x_real.cuda()
                z_input = z_input.cuda()

            x_fake = networks['G'](z_input)

            if i == 0:
                # G update
                optimizers['G'].zero_grad()

                # G forward
                logit_d_fake = networks['D'](x_fake)
                loss_g = F.binary_cross_entropy(logit_d_fake, ones)

                # G backward
                loss_g.backward()
                optimizers['G'].step()
                if is_main:
                    accumulate(networks['G_running'], networks['G'].module)

                # AM update
                am_loss_g.update(loss_g.item(), x_real.size(0))

            # D update
            optimizers['D'].zero_grad()
            x_real.requires_grad_()

            # D real forward
            logit_d_real = networks['D'](x_real)
            loss_d_real = F.binary_cross_entropy(logit_d_real, ones)

            # D real regularization - 0-GP
            loss_gp = 5.0 * compute_zero_gp(logit_d_real, x_real).mean()

            # D fake forward
            logit_d_fake = networks['D'](x_fake.detach())
            loss_d_fake = F.binary_cross_entropy(logit_d_fake, zeros)

            # D step
            loss_d = loss_d_real + loss_d_fake + loss_gp
            loss_d.backward()
            optimizers['D'].step()

            # AM update
            am_loss_d.update(loss_d.item(), x_real.size(0))
            am_mean_r.update(logit_d_real.mean().item(), x_real.size(0))
            am_mean_f.update(logit_d_fake.mean().item(), x_real.size(0))

        if t % args.log_step == 0 and is_main:
            t_train.set_description('Epoch: [{}/{}], '
                                    'Loss: '
                                    'D[{loss_d.avg:.3f}] '
                                    'G[{loss_g.avg:.3f}] '
                                    'Fm[{mean_f.avg:.3f}] '
                                    'Rm[{mean_r.avg:.3f}]'
                                    ''.format(epoch,
                                              args.epochs,
                                              loss_d=am_loss_d,
                                              loss_g=am_loss_g,
                                              mean_f=am_mean_f,
                                              mean_r=am_mean_r)
                                    )


def validate(val_loader, networks, epoch, args, is_main=False):
    networks['G_running'].eval()

    subFID_total = 1000
    batch_size = args.batch_size

    with torch.no_grad():
        metric = FID(batch_size=batch_size, gpu=args.gpu)
        metric.load_model()

        trange = tqdm.trange(subFID_total // batch_size + 1)

        trange.set_description('Epoch: [{}/{}], '
                               'Metric: '
                               'FID1K [{fid_score:.3f}] '
                               ''.format(epoch,
                                         args.epochs,
                                         fid_score=0)
                               )
        for i, _ in enumerate(trange):
            z_val = torch.randn(args.batch_size, args.latent_size)
            if args.gpu is not None:
                z_val = z_val.cuda(args.gpu)

            x_fake = (networks['G_running'](z_val) + 1) / 2

            metric.aggregate_activations(x_fake)

            if i == subFID_total // batch_size:
                vutils.save_image(x_fake.cpu(),
                                  os.path.join(args.res_dir, "fake_{}.jpg".format(epoch)),
                                  normalize=True,
                                  nrow=int(np.sqrt(args.batch_size)))

                metric.free_model()
                metric.calculate_statistics()
                fid_score = metric.measure_fid("./metrics/celeba_anno_val_full.npz")
                trange.set_description('Epoch: [{}/{}], '
                                       'Metric: '
                                       'FID1K [{fid_score:.3f}] '
                                       ''.format(epoch,
                                                 args.epochs,
                                                 fid_score=fid_score)
                                       )