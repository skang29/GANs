""""
Layers / linear layers under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
from ..normalizations import spectral_norm
NCCL_FLAG = os.environ.get('nccl_multigpu_env')


def linear(input_, output_size, name='linear', bias_init=0.0, sn=False, with_w=False, tower_config=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable(name="w",
                            shape=[shape[1], output_size],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name="b",
                            shape=[output_size],
                            initializer=tf.constant_initializer(bias_init))

        if sn:
            y = tf.matmul(input_, spectral_norm(w, tower_config=tower_config)) + b

        else:
            y = tf.matmul(input_, w) + b

        if with_w:
            return y, w, b
        else:
            return y
