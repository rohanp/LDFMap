from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize([Extension("calcRMSD", 
    					sources = ["calcRMSD.pyx"],
    					include_dirs = [np.get_include(), "/Users/rohanp/newLDFMap/RMSD/include"],
    					language="c++",
    					libraries = ["lapack", "blas"],
    					library_dirs = ["/Users/rohanp/newLDFMap/RMSD/include"],
    					extra_compile_args = ["-I /Users/rohanp/newLDFMap/RMSD/include"],
    					#extra_link_args = ["-L~/newLDFMap/RMSD/include"]
    					)])
   							
   	)