import tensorflow as tf
import numpy as np

from .ops import *
from .utils import nested_dict

from .generator import decoder
from .discriminator import discriminator
from .words import *

from tensorflow.contrib.nccl.python.ops import nccl_ops
nccl_ops._maybe_load_nccl_ops_so()


class Network(object):
    def __init__(self,
                 batch_size=64,
                 size=128,
                 gf_dim=64,
                 df_dim=64,
                 gf_mult=8,
                 df_mult=8,
                 name="Network",
                 reuse=False,
                 is_training=True,
                 tower_config=None):
        self.batch_size = batch_size
        self.size = size
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gf_mult = gf_mult
        self.df_mult = df_mult

        self.name = name
        self.reuse = reuse
        self.is_training = is_training

        self.tower_config = tower_config

        # Network params
        self.C_gradient = 10

    def build_network(self, z, x):
        with tf.variable_scope(self.name):
            if self.is_training:
                self._build_train_network(z, x)
            else:
                self.reuse = True
                self._build_test_network(z, x)

    def _build_train_network(self, z, x):
        self.z = z
        self.x = x

        # Generator
        self.y = decoder(z_=self.z,
                         batch_size=self.batch_size,
                         size=self.size,
                         gf_dim=self.gf_dim,
                         name="generator",
                         reuse=self.reuse,
                         tower_config=self.tower_config)

        # Discriminator
        self.D = nested_dict()
        self.D[real][sigmoid], self.D[real][logits] = discriminator(image_=self.x,
                                                                    batch_size=self.batch_size,
                                                                    df_dim=self.df_dim,
                                                                    name="discriminator",
                                                                    reuse=self.reuse,
                                                                    tower_config=self.tower_config)

        self.D[fake][sigmoid], self.D[fake][logits] = discriminator(image_=self.y,
                                                                    batch_size=self.batch_size,
                                                                    df_dim=self.df_dim,
                                                                    name="discriminator",
                                                                    reuse=True,
                                                                    tower_config=self.tower_config)

        # Loss
        # Discriminator loss
        # Adversarial loss
        self.dLoss = nested_dict()
        self.dLoss[real] = loss_iterator(self.D[real][logits], tf.ones_like)
        self.dLoss[fake] = loss_iterator(self.D[fake][logits], tf.zeros_like)

        # Gradient Penalty on real distribution
        # self.gp_loss = gradient_penalty(self.x, self.D[real][logits], self.tower_config)

        self.dPureLoss = self.dLoss[real] + self.dLoss[fake]
        self.d_loss = self.dPureLoss #+ self.gp_loss


        # Generator loss
        # Adversarial loss
        self.gLoss = nested_dict()
        self.gLoss[fake] = loss_iterator(self.D[fake][logits], tf.ones_like)

        self.g_loss = self.gPureLoss = self.gLoss[fake]

    def _build_test_network(self, z, x):
        self.z = z
        self.x = x

        # Generator
        self.y = decoder(z_=self.z,
                         batch_size=self.batch_size,
                         size=self.size,
                         gf_dim=self.gf_dim,
                         name="generator",
                         reuse=self.reuse,
                         tower_config=self.tower_config,
                         is_training=self.is_training)
