import torch.nn as nn


# class Reshape(nn.Module):
#     def __init__(self, *args):
#         super(Reshape, self).__init__()
#         self.shape = args
#
#     def forward(self, x):
#         return x.view(self.shape)


def spectral_norm(module, sn=False):
    if sn:
        return nn.utils.spectral_norm(module)
    else:
        return module

