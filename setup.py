from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

# usage: python setup.py build_ext --inplace
# you will have to change include_dirs, library_dirs, and extra_compile_args
# to the absolute path of the Include directory in your system

setup(
    ext_modules = cythonize([Extension("LDFMap_debug", 
    					sources = ["LDFMap_debug.pyx"],
    					include_dirs = [np.get_include(), "/Users/rohanp/LDFMap/include"],
    					language="c++",
    					libraries = ["lapack", "blas"],
    					library_dirs = ["/Users/rohanp/LDFMap/Include"],
    					extra_compile_args = ["-I /Users/rohanp/LDFMap/Include"],
    					)])
   							
   	)

