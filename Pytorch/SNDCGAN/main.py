import os
import argparse
import time
import warnings
from datetime import datetime

import torch
import torch.multiprocessing as mp

from parallel_training_model import main_worker

from utils import *

parser = argparse.ArgumentParser(description='PyTorch GAN Training')
parser.add_argument('--img_dir', default='/home/Databases/CelebA/images', help='path to dataset')
parser.add_argument('--model_name', type=str, default='GAN', help='model name')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Total batch size - e.g) num_gpus = 2 , batch_size = 128 then, effectively, 64')
parser.add_argument('--latent_size', type=int, default=128, help='Length of latent vector')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay (default: 0, general: 1e-4)', dest='weight_decay')
parser.add_argument('--log_step', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('--load_model', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--validation', dest='validation', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8888', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpus', default='0', type=str,
                    help='GPU ids to use.')
parser.add_argument('--multiprocessing-distributed', action='store_false',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# os.environ['NCCL_DEBUG'] = 'INFO'

def main():
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        # warnings.warn('You have chosen a specific GPU. This will completely '
        #               'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.load_model is None:
        args.model_name = '{}_{}'.format(args.model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        args.model_name = args.load_model

    makedirs('./logs')
    makedirs('./results')

    args.log_dir = os.path.join('./logs', args.model_name)
    args.res_dir = os.path.join('./results', args.model_name)

    makedirs(args.log_dir)
    makedirs(args.res_dir)

    formatted_print('Total Number of GPUs:', ngpus_per_node)
    formatted_print('Total Number of Workers:', args.workers)
    formatted_print('Effective Batch Size:', args.batch_size * ngpus_per_node)

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__=="__main__":
    main()
