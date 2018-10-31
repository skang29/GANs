from tensorflow.contrib.nccl.python.ops import nccl_ops
# Force including NCCL Ops
nccl_ops._maybe_load_nccl_ops_so()

from .activations import *
from .blocks import *
from .convolutions import *
from .linears import *
from .losses import *
from .normalizations import *
from .tfthings import *

from .customs import *


VERSION_INFO = \
"""----------------------------
| Ops
| Version: 2.0.0
| Change log
|   * Version established.
|   * Parallel mode added.
| Modified date: 2018.10.30.
----------------------------"""
print(VERSION_INFO)
