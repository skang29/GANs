""""
Convolution layers under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
from .normalizations import spectral_norm
NCCL_FLAG = os.environ.get('nccl_multigpu_env')


if NCCL_FLAG == 'true':
    def conv2d(input_, output_dim, kernel=(5, 5), strides=(2, 2), sn=False, name="conv2d", tower_config=None):
        with tf.variable_scope(name):
            w = tf.get_variable(name="w",
                                shape=[kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name='b',
                                shape=[output_dim],
                                initializer=tf.constant_initializer(0.0))

            if sn:
                conv = tf.nn.conv2d(input_, spectral_norm(w, tower_config=tower_config),
                                    strides=[1, strides[0], strides[1], 1],
                                    padding='SAME')

            else:
                conv = tf.nn.conv2d(input_, w,
                                    strides=[1, strides[0], strides[1], 1],
                                    padding='SAME')

            return tf.nn.bias_add(conv, b)


    def conv2d_transpose(input_,
                         output_shape,
                         kernel=(4, 4),
                         strides=(2, 2),
                         sn=False,
                         name="conv2d_transpose",
                         with_w=False,
                         tower_config=None):
        with tf.variable_scope(name):
            # filter : (height, width, output_channels, in_channels)
            w = tf.get_variable(name="w",
                                shape=[kernel[0], kernel[1], output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name="b",
                                shape=[output_shape[-1]],
                                initializer=tf.constant_initializer(0.0))

            if sn:
                conv_tp = tf.nn.conv2d_transpose(input_, spectral_norm(w, tower_config=tower_config),
                                                 output_shape=output_shape,
                                                 strides=[1, strides[0], strides[1], 1])

            else:
                conv_tp = tf.nn.conv2d_transpose(input_, w,
                                                 output_shape=output_shape,
                                                 strides=[1, strides[0], strides[1], 1])

            conv_tp = tf.reshape(tf.nn.bias_add(conv_tp, b), conv_tp.get_shape())

            if with_w:
                return conv_tp, w, b
            else:
                return conv_tp

else:
    def conv2d(input_, output_dim, kernel=(5, 5), strides=(2, 2), sn=False, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable(name="w",
                                shape=[kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name='b',
                                shape=[output_dim],
                                initializer=tf.constant_initializer(0.0))

            if sn:
                conv = tf.nn.conv2d(input_, spectral_norm(w),
                                    strides=[1, strides[0], strides[1], 1],
                                    padding='SAME')

            else:
                conv = tf.nn.conv2d(input_, w,
                                    strides=[1, strides[0], strides[1], 1],
                                    padding='SAME')

            return tf.nn.bias_add(conv, b)


    def conv2d_transpose(input_,
                         output_shape,
                         kernel=(4, 4),
                         strides=(2, 2),
                         sn=False,
                         name="conv2d_transpose",
                         with_w=False):
        with tf.variable_scope(name):
            # filter : (height, width, output_channels, in_channels)
            w = tf.get_variable(name="w",
                                shape=[kernel[0], kernel[1], output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name="b",
                                shape=[output_shape[-1]],
                                initializer=tf.constant_initializer(0.0))

            if sn:
                conv_tp = tf.nn.conv2d_transpose(input_, spectral_norm(w),
                                                 output_shape=output_shape,
                                                 strides=[1, strides[0], strides[1], 1])

            else:
                conv_tp = tf.nn.conv2d_transpose(input_, w,
                                                 output_shape=output_shape,
                                                 strides=[1, strides[0], strides[1], 1])

            conv_tp = tf.reshape(tf.nn.bias_add(conv_tp, b), conv_tp.get_shape())

            if with_w:
                return conv_tp, w, b
            else:
                return conv_tp
