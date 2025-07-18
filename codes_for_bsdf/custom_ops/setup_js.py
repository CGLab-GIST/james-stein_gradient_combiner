
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
	name='js',
	ext_modules=[CUDAExtension('js_cpp', sources=['james_stein/js.cpp', 'james_stein/cuda_js.cu'], extra_compile_args={'cxx' : [], 'nvcc' : ['-arch', 'compute_70']})],
	py_modules=["james_stein/js"],
	cmdclass={
		'build_ext': BuildExtension
	}
)
