""""
Application blocks under tensorflow environment.
Supports NCCL multi-gpu environment.
To activate the environment, use code below in your
main.py.
>> os.environ['nccl_multigpu_env'] = 'true'
"""
__version__ = "1.0.0"

import os
import tensorflow as tf
from .convolutions import conv2d
from .activations import relu
from .normalizations import group_norm
NCCL_FLAG = os.environ.get('nccl_multigpu_env')


if NCCL_FLAG == 'true':
    def residual_block(input_layer,
                       output_channel=None,
                       hidden_channel=None,
                       normalization=False,
                       sn=False,
                       name="res_block",
                       tower_config=None):
        """
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param hidden_channel: int.
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.

        ### Important ###
        This residual block is from Which Training Method ....
        """

        UserWarning("This residual block is from Which Training Method ... . ")

        with tf.variable_scope(name) as scope:
            input_channel = input_layer.get_shape().as_list()[-1]

            if output_channel is None:
                output_channel = input_channel

            if hidden_channel is None:
                hidden_channel = min(input_channel, output_channel)

            x = nets = input_layer
            if normalization: nets = group_norm(nets, name="gn0", tower_config=tower_config)
            nets = relu(nets, tower_config = tower_config)

            nets = conv2d(input_=nets,
                          output_dim=hidden_channel,
                          kernel=(3, 3),
                          strides=(1, 1),
                          name="conv_h0",
                          sn=sn,
                          tower_config = tower_config)

            if normalization: nets = group_norm(nets, name="gn1", tower_config = tower_config)
            nets = relu(nets, tower_config = tower_config)

            dx = conv2d(input_=nets,
                        output_dim=output_channel,
                        kernel=(3, 3),
                        strides=(1, 1),
                        name="conv_h1",
                        sn=sn,
                        tower_config=tower_config)

            if output_channel != input_channel:
                x = conv2d(input_=x,
                           output_dim=output_channel,
                           kernel=(3, 3),
                           strides=(1, 1),
                           name="conv_s",
                           sn=sn,
                           tower_config=tower_config)

            ret = x + dx
            return ret

else:
    def residual_block(input_layer, output_channel=None, hidden_channel=None, normalization=False, sn=False,
                       name="res_block"):
        """
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param hidden_channel: int.
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.

        ### Important ###
        This residual block is from Which Training Method ....
        """

        UserWarning("This residual block is from Which Training Method ... . ")

        with tf.variable_scope(name) as scope:
            input_channel = input_layer.get_shape().as_list()[-1]

            if output_channel is None:
                output_channel = input_channel

            if hidden_channel is None:
                hidden_channel = min(input_channel, output_channel)

            x = nets = input_layer
            if normalization: nets = group_norm(nets, name="gn0")
            nets = relu(nets)

            nets = conv2d(input_=nets,
                          output_dim=hidden_channel,
                          kernel=(3, 3),
                          strides=(1, 1),
                          name="conv_h0",
                          sn=sn)

            if normalization: nets = group_norm(nets, name="gn1")
            nets = relu(nets)

            dx = conv2d(input_=nets,
                        output_dim=output_channel,
                        kernel=(3, 3),
                        strides=(1, 1),
                        name="conv_h1",
                        sn=sn)

            if output_channel != input_channel:
                x = conv2d(input_=x,
                           output_dim=output_channel,
                           kernel=(3, 3),
                           strides=(1, 1),
                           name="conv_s",
                           sn=sn)

            ret = x + dx
            return ret

