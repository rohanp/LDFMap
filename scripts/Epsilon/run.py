#!/usr/bin/env
""" Compile+Run Script for calcMarkov.pyx """
__author__ = "Rohan Pandit"

import time
import numpy as np
import pyximport 
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

import calcEpsilon_nogil
#import test_cython

t0 = time.time()

numModels = 500

r = np.random.rand(numModels, numModels)
RMSD = (np.tril(r) + np.tril(r).T) * (1 - np.eye(r.shape[0]))*4

print(RMSD.shape)

maxEpsilon = np.max(RMSD)
possibleEpsilons = np.array([(3./7.)*maxEpsilon, (1./2.)*maxEpsilon, (4./7.)*maxEpsilon])
epsilons = []

for i in range(numModels):
	print("######## epsilon {0} ########".format(i))
	epsilons.append(calcEpsilon_nogil.calcEpsilon(i, RMSD, possibleEpsilons, 0.3))


print(epsilons)

print( time.time() - t0 )