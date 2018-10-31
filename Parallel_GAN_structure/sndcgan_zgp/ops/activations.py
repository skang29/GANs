""""
Activations under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
NCCL_FLAG = os.environ.get('nccl_multigpu_env')


if NCCL_FLAG == 'true':
    def lrelu(x, leak=0.2, tower_config=None):
        return tf.maximum(x, leak * x)


    def relu(x, tower_config=None):
        return tf.maximum(x, 0)

else:
    def lrelu(x, leak=0.2):
        return tf.maximum(x, leak * x)


    def relu(x):
        return tf.maximum(x, 0)
