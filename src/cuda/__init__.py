"""
CUDA-accelerated Grassmann operations.

To build: cd src/cuda && python setup.py install
"""

try:
    import grassmann_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    grassmann_cuda = None

from .grassmann_fused import (
    FusedGrassmannMixing,
    FusedGrassmannBlock,
    GrassmannGPTFused,
)

__all__ = [
    'CUDA_AVAILABLE',
    'grassmann_cuda',
    'FusedGrassmannMixing',
    'FusedGrassmannBlock',
    'GrassmannGPTFused',
]
