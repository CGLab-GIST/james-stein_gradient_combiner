from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
	name='filter',
	ext_modules=[CUDAExtension('filter_cpp', sources=['filter/filter.cpp', 'filter/cuda_filter.cu'], extra_compile_args={'cxx' : [], 'nvcc' : ['-arch', 'compute_70']})],
	py_modules=["filter/filter"],
	cmdclass={
		'build_ext': BuildExtension
	}
)
