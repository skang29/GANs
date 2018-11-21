""""
Normalizations / layer_norm under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
NCCL_FLAG = os.environ.get('nccl_multigpu_env')

__all__ = ['layer_norm']


# Layer Normalization does not need cross-batch computation.
def layer_norm(x, name='layer_norm', tower_config=None):
    """Doesn't need cross-batch computation."""
    with tf.variable_scope(name):
        return tf.contrib.layers.layer_norm(x)
