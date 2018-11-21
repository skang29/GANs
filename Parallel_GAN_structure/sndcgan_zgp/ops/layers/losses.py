""""
Layers / loss calculation blocks under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
from tensorflow.contrib.nccl.ops import gen_nccl_ops
NCCL_FLAG = os.environ.get('nccl_multigpu_env')


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


if NCCL_FLAG == 'true':
    # NCCL multi-gpu environment.
    def gradient_penalty(data, logits, tower_config, coefficient=10.0, name="gradient_penalty"):
        nccl_name = "NCCL" if not tower_config.is_test else "NCCL_TEST"

        with tf.variable_scope(name):
            gradient = tf.gradients(tf.reduce_sum(logits), [data])[0]
            slope_square = tf.reduce_mean(tf.reduce_sum(tf.square(gradient), reduction_indices=[1, 2, 3]))

            shared_name = slope_square.name. \
                replace(tower_config.name, tower_config.prefix.format(nccl_name))

            penalty = gen_nccl_ops.nccl_all_reduce(
                input=slope_square,
                reduction="sum",
                num_devices=tower_config.num_devices,
                shared_name=shared_name
            ) / (1.0 * tower_config.num_devices * coefficient)

            return penalty

else:
    # None NCCL multi-gpu environment.
    def gradient_penalty(data, logits, coefficient=10.0, name="gradient_penalty", tower_config=None):
        with tf.variable_scope(name):
            gradient = tf.gradients(tf.reduce_sum(logits), [data])[0]
            slope_square = tf.reduce_mean(tf.reduce_sum(tf.square(gradient), reduction_indices=[1, 2, 3]))

            penalty = tf.reduce_mean(slope_square) * coefficient

            return penalty
