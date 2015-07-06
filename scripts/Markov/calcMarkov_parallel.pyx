#encoding: utf-8
#cython: wraparound=False, boundscheck=False, cdivision=True
#cython: profile=False, nonecheck=False
#filename: diffusionMap.py
#by Rohan Pandit

from __future__ import division
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.math cimport sqrt, exp

def main(RMSD, epsilonArray, N, num_threads):
	P = calcMarkovMatrix(RMSD, epsilonArray, N, num_threads)
	return P

cdef double[:,:] calcMarkovMatrix(double[:,:] RMSD, double[:] epsilonArray, int N, int num_t):	

	cdef int i, j
	cdef double[:] D = np.zeros(N)
	cdef double[:] Dtilda = np.zeros(N)
	cdef double[:,:] K = np.zeros((N,N))
	cdef double[:,:] Ktilda = np.zeros((N,N))
	cdef double[:,:] P = np.zeros((N,N))

	for i in prange(N, nogil=True, schedule="static", num_threads = num_t):
		for j in range(N):
			K[i][j] = exp((-RMSD[i][j]*RMSD[i][j])/(2*epsilonArray[i]*epsilonArray[j]))
			D[i] += K[i][j]

			Ktilda[i][j] = K[i][j]/sqrt(D[i]*D[j])
			Dtilda[i] += Ktilda[i][j]

			P[i][j] = Ktilda[i][j]/Dtilda[i]

	return P