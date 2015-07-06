import pstats,cProfile
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})

import calcEpsilon_gil
#import test_cython

#cProfile.runctx("test_cython.main()", globals(), locals(), "Profile.prof")
cProfile.runctx("calcEpsilon_gil.main(500)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof", stream = open("profile.txt", 'w'))
s.strip_dirs().sort_stats("time").print_stats()