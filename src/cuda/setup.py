from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='grassmann_cuda',
    ext_modules=[
        CUDAExtension(
            name='grassmann_cuda',
            sources=['grassmann_kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '--expt-relaxed-constexpr',
                    '--use_fast_math',
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_90,code=sm_90',  # H100
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
