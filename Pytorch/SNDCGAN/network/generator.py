import torch.nn as nn
import numpy as np
from .ops import spectral_norm

leak = 0.2

minimum_image_size = 16


class Generator(nn.Module):
    def __init__(self, num_gpus=1, img_size=64, latent_size=128, nf=64, nc=3, sn=False):
        super(Generator, self).__init__()
        self.num_gpus = num_gpus
        self.nf = nf
        self.nc = nc
        self.img_size = img_size
        self.latent_size = latent_size
        self.max_nf = 512
        self.sn = sn

        num_layers = int(np.log2(img_size / minimum_image_size))

        self.nf_init = min(self.max_nf, self.nf * int(2 ** num_layers))

        self.fc = nn.Sequential(*init_stage(latent_size,
                                            (minimum_image_size ** 2) * self.nf_init,
                                            sn=sn))

        model_list = list()

        for i in range(num_layers):
            nf_in = min(self.max_nf, self.nf * int(2 ** (num_layers - i)))
            nf_out = min(self.max_nf, self.nf * int(2 ** (num_layers - 1 - i)))

            model_list += repeat_stage(nf_in, nf_out, sn=sn)

        model_list += final_stage(self.nf, self.nc, sn=sn)

        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        m = self.fc(x)
        m = m.view(m.size(0), self.nf_init, minimum_image_size, minimum_image_size)
        m = self.model(m)

        return m


def init_stage(nz, nf, sn=False):
    model = [
        spectral_norm(nn.Linear(nz, nf), sn=sn),
        nn.ReLU()
    ]

    return model


def repeat_stage(nf_in, nf_out, sn=False):
    model = [
        spectral_norm(nn.ConvTranspose2d(nf_in, nf_out, 4, 2, 1), sn=sn),
        nn.BatchNorm2d(nf_out),
        nn.ReLU()
    ]

    return model


def final_stage(nf_in, nf_out, sn=False):
    model = [
        spectral_norm(nn.Conv2d(nf_in, nf_out, 3, 1, 1),sn=sn),
        nn.Tanh()
    ]

    return model
