#!/usr/bin/env
""" Compile+Run Script for calcMarkov.pyx """
__author__ = "Rohan Pandit"

import time
import numpy as np
import pyximport 
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import calcMarkov_parallel
import calcMarkov_nogil
import calcMarkov
import calcMarkov_python
import matplotlib.pyplot as pyplot

num_models = 1000
num_threads = 8
trials = 10

RMSD = np.random.rand( num_models, num_models)*5
epsilonArray = np.random.rand(num_models)*3 + 0.5

for i in range(trials):
	t0 = time.time()
	P = calcMarkov_nogil.main(RMSD, epsilonArray, num_models)
	nogil_time = time.time() - t0

	t0 = time.time()
	P = calcMarkov_parallel.main(RMSD, epsilonArray, num_models, num_threads)
	parallel_time =  time.time() - t0

	t0 = time.time()
	P = calcMarkov.main(RMSD, epsilonArray, num_models)
	cython_time =  time.time() - t0

	t0 = time.time()
	P = calcMarkov_python.main(RMSD, epsilonArray, num_models)
	python_time = time.time() - t0

fig, ax = pyplot.subplots()
ax.bar([1, 2, 3, 4], [python_time/trials, cython_time/trials, nogil_time/trials, parallel_time/trials], 0.5)
ax.set_xticks([0.75, 1.25, 2.25, 3.25, 4.25, 4.5])
ax.set_ylabel("Time (seconds)")
ax.set_xticklabels(("", "Python", "Cython", "Cython (No Gil)", "Parallel Cython (No Gil)", ""))
pyplot.show()