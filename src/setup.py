from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import os

# usage: python setup.py build_ext --inplace
# you will have to change include_dirs, library_dirs, and extra_compile_args
# to the absolute path of the Include directory in your system

#os.environ["CC"] = "g++-5"
#os.environ["CXX"] = "g++-5"

setup(
    author = "Rohan Pandit",
    url='https://www.github.com/rohanp/LDFMap',
    ext_modules = cythonize([Extension("LDFMap", 
    					sources = ["LDFMap.pyx"],
    					include_dirs = [np.get_include(), "/Users/rohanp/LDFMap/src/include"],
    					language="c++",
    					libraries = ["lapack", "blas"],
    					library_dirs = ["/Users/rohanp/LDFMap/src/include"],
    					extra_compile_args = ["-I /Users/rohanp/LDFMap/src/include", "-I /usr/local/include"],
                        #extra_link_args = ['-fopenmp']
    					)])
   							
   	)

