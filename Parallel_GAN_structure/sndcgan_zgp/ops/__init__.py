from tensorflow.contrib.nccl.python.ops import nccl_ops
from .customs import *
from .nccl_utils import *
from .layers import *
from .core import *
from .blocks import *
from .normalizations import *


# Force including NCCL Ops
nccl_ops._maybe_load_nccl_ops_so()


VERSION_INFO = \
"""----------------------------
| Ops
| Version: 2.0.0
| Change log
|   * Version established.
|   * Parallel mode added.
|   * New ops structure established.
| Modified date: 2018.11.21.
----------------------------"""
print(VERSION_INFO)
