import torch.nn as nn
import numpy as np
from .ops import spectral_norm

leak = 0.2

minimum_image_size = 16


class Discriminator(nn.Module):
    def __init__(self, num_gpus=1, img_size=64, nf=64, nc=3, sn=False):
        super(Discriminator, self).__init__()
        self.num_gpus = num_gpus
        self.nf = nf
        self.nc = nc
        self.img_size = img_size
        self.max_nf = 512
        self.sn = sn

        num_layers = int(np.log2(img_size / minimum_image_size))

        model_list = list()
        model_list += init_stage(self.nc, self.nf, sn=sn)

        for i in range(num_layers):
            nf_in = min(self.max_nf, self.nf * int(2 ** i))
            nf_out = min(self.max_nf, self.nf * int(2 ** (i+1)))

            model_list += repeat_stage(nf_in, nf_out, sn=sn)

        self.model = nn.Sequential(*model_list)

        self.nf_final = min(self.max_nf, self.nf * int(2 ** num_layers))
        self.fc = nn.Sequential(*final_stage((minimum_image_size ** 2) * self.nf_final, 1, sn=sn))

    def forward(self, x):
        m = self.model(x)
        m = m.view(m.size(0), -1)
        m = self.fc(m)

        return m


def init_stage(nc, nf, sn=False):
    model = [
        spectral_norm(nn.Conv2d(nc, nf, 3, 1, 1), sn=sn),
        nn.LeakyReLU(leak)
    ]

    return model


def repeat_stage(nf_in, nf_out, sn=False):
    model = [
        spectral_norm(nn.Conv2d(nf_in, nf_out, 4, 2, 1), sn=sn),
        nn.LeakyReLU(leak),

        spectral_norm(nn.Conv2d(nf_out, nf_out, 3, 1, 1), sn=sn),
        nn.LeakyReLU(leak)
    ]

    return model


def final_stage(nf_in, nf_out, sn=False):
    model = [
        spectral_norm(nn.Linear(nf_in, nf_out), sn=sn),
        nn.Sigmoid()
    ]

    return model
