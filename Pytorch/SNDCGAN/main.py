import os
import time
from datetime import datetime
import argparse
import json

import torch
import torch.multiprocessing as mp

from utils import makedirs, formatted_print
from parallel_training_model import main_worker

parser = argparse.ArgumentParser(description='Pytorch COCOGAN')
parser.add_argument('--dataset_dir', type=str, default='/home/Databases/CelebA/CELEBA_ANNO/', help='Dataset path.')
parser.add_argument('--model_name', type=str, default='COCOGAN', help='Model name')
parser.add_argument('--epochs', default=200, type=int, help='Number of total epochs to run')
parser.add_argument('--steps', default=500, type=int, help='Steps per epoch')
parser.add_argument('--ttur_d', default=4, type=int, help='TTUR: N per 1 generator update')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Total batch size (e.g., num_gpus = 2 , batch_size = 128 then, effectively, 64 per GPU)')

parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay (default: 0, general: 1e-4)', dest='weight_decay')

parser.add_argument('--log_step', default=1, type=int, help='Log step')
parser.add_argument('--val_epoch', default=20, type=int, help='Validate every N epochs.')
parser.add_argument('--validation', dest='validation', action='store_true', help='evaluate model on validation set')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--load_model', default=None, type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:8888',
                    type=str, help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

parser.add_argument('--gpus', default='0', type=str, help='GPU ids to use.')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')

parser.add_argument('--latent_size', type=int, default=128, help='Length of latent vector')


# with open('commandline_args.txt', 'w') as f:
#     json.dump(args.__dict__, f, indent=2)


def main():
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

        _num_gpus = len(args.gpus.split(","))
        assert _num_gpus == torch.cuda.device_count(), \
            "Number of Visible GPUs ({}) is different from GPU option argument ({}).".format(_num_gpus, args.gpus)

    ngpus_per_node = torch.cuda.device_count()

    assert args.batch_size % ngpus_per_node == 0, \
        "Batch size ({}) must be divided by the number of GPUs per node ({}).".format(args.batch_size, ngpus_per_node)
    args.gpu_batch_size = args.batch_size // ngpus_per_node

    args.distributed = args.world_size > 1 or ngpus_per_node > 1

    if args.load_model is None:
        args.sub_model_name = "{}_{}".format(args.model_name, datetime.now().strftime("%Y%m%d%H%M%S"))
    else:
        args.sub_model_name = os.path.split(args.load_model)[1]

    args.base_path = os.path.join("./Container", args.model_name, args.sub_model_name)
    args.log_dir = os.path.join(args.base_path, "logs")
    args.res_dir = os.path.join(args.base_path, "results")
    args.ckpt_dir = os.path.join(args.base_path, "ckpt")

    makedirs(args.log_dir)
    makedirs(args.res_dir)
    makedirs(args.ckpt_dir)

    formatted_print('Total number of GPUs:', ngpus_per_node)
    formatted_print('Total number of workers:', args.workers)
    formatted_print('Total batch size:', args.batch_size)

    with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        main_worker(0, ngpus_per_node, args)


if __name__=="__main__":
    main()

