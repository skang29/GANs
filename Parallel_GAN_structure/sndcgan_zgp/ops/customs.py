""""
Custom layers under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
from .layers.losses import sigmoid_cross_entropy_with_logits
NCCL_FLAG = os.environ.get('nccl_multigpu_env')


def network_sum(network_list, lambda_function):
    return sum([lambda_function(network) for network in network_list])


def network_mean(network_list, lambda_function):
    return sum([lambda_function(network) for network in network_list]) / len(network_list)


def loss_iterator(logits, fitting):
    loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(
            logits,
            fitting(logits))
    )

    return loss
