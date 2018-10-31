""""
Loss calculation blocks under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
NCCL_FLAG = os.environ.get('nccl_multigpu_env')


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
