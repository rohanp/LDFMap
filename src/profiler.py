import pstats,cProfile
import numpy as np

import LDFMap_debug

#cProfile.runctx("test_cython.main()", globals(), locals(), "Profile.prof")
cProfile.runctx("LDFMap_debug.main('Input/1000_modified_Met-Enk_AMBER.pdb', 40, 999)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof", stream = open("Output/profile.txt", 'w'))
s.strip_dirs().sort_stats("time").print_stats()