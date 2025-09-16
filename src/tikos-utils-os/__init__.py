"""
CUDA Batched Tensor Operations (tikos-utils-os)

A PyPI package that leverages PyCUDA for memory-efficient, tensor operations with unequal dimensions (multiplication and division) on NVIDIA GPUs.
It processes the element-wise operations in batches, ensuring large tensors in slices to keep VRAM usage low.
"""
from .tutils import multiply, divide

__version__ = "0.1.0"