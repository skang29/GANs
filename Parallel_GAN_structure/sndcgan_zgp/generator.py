import tensorflow as tf
import numpy as np

from .ops import *
from .utils import print_info


def normalization_function(nets, name="norm", tower_config=None):
    return batch_norm(nets, name=name, tower_config=tower_config)


def decoder(z_, batch_size, size, gf_dim, name="decoder", sn=False, reuse=False, debug=False, tower_config=None, is_training=False):
    with tf.variable_scope("type__generator"):
        with tf.variable_scope(name, reuse=reuse):
            num_layers = int(np.log2(size / 4)) - 1

            if debug: print_info("Input", z_)
            nets = init_stage(z_, batch_size, gf_dim * int(2 ** num_layers), sn=sn, tower_config=tower_config, is_training=is_training)
            if debug: print_info("Init", nets)
            # nets: [batch_size, 4, 4, gf_dim * ]

            for i in range(num_layers):
                conv_dims = gf_dim * int(2 ** (num_layers - i))
                nets = repeat_stage(nets,
                                    batch_size,
                                    conv_dims,
                                    name="repeat_stage_{}".format(i),
                                    sn=sn,
                                    tower_config=tower_config,
                                    is_training=is_training)
                if debug: print_info("Repeat_{}".format(i), nets)

            # nets: [batch_size, size / 2, size / 2, gf_dim]

            nets = final_stage(nets, batch_size, gf_dim, sn=sn, tower_config=tower_config, is_training=is_training)
            if debug: print_info("Final", nets)
            # nets: [batch_size, size, size, 3]

            return tf.nn.tanh(nets)


def init_stage(z_, batch_size, conv_dims, name="init_stage", sn=False, tower_config=None, is_training=False):
    with tf.variable_scope(name):
        nets = linear(z_, 4 * 4 * conv_dims, sn=sn, tower_config=tower_config)
        nets = tf.reshape(nets, [batch_size, 4, 4, conv_dims])

        return nets


def repeat_stage(nets, batch_size, conv_dims, name="repeat_stage", sn=False, tower_config=None, is_training=False):
    with tf.variable_scope(name):
        nets = up_block(nets, batch_size, conv_dims, sn=sn, tower_config=tower_config)

        return nets


def final_stage(nets, batch_size, conv_dims, name="final_stage", sn=False, tower_config=None, is_training=False):
    with tf.variable_scope(name):
        nets = up_block(nets, batch_size, conv_dims, sn=sn, tower_config=tower_config)
        nets = batch_norm(nets, tower_config=tower_config)
        nets = relu(nets, tower_config=tower_config)
        nets = conv2d(nets, 3, kernel=(3, 3), strides=(1, 1), sn=sn, tower_config=tower_config)

        return nets


def up_block(x, batch_size, conv_dims, name="upblock", sn=False, tower_config=None, is_training=False):
    with tf.variable_scope(name):
        size = x.get_shape().as_list()
        nets = residual_block(x, output_channel=conv_dims, normalization=normalization_function, sn=sn, tower_config=tower_config)
        nets = tf.image.resize_images(nets, [size[1] * 2, size[2] * 2], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return nets
