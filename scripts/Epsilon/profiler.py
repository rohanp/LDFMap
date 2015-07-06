import pstats,cProfile
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})

#import calcEpsilon_nogil
import test_cython

cProfile.runctx("test_cython.main()", globals(), locals(), "Profile.prof")
#cProfile.runctx("calcEpsilon.main('blah', 500, 0.3)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()