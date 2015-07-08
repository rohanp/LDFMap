#encoding: utf-8
#cython: wraparound=False, boundscheck=False, cdivision=True
#cython: profile=False, nonecheck=False
#filename: diffusionMap.py
#by Rohan Pandit

from __future__ import division
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, exp

def main(RMSD, epsilonArray, N):
	P = calcMarkovMatrix(RMSD, epsilonArray, N)
	return P

cdef double[:,:] calcMarkovMatrix(double[:,:] RMSD, double[:] epsilonArray, int N):	
	cdef:
		int i, j
		double[:] D = np.zeros(N)
		double[:] Dtilda = np.zeros(N)
		double[:,:] K = np.zeros((N,N))
		double[:,:] Ktilda = np.zeros((N,N))
		double[:,:] P = np.zeros((N,N))

	with nogil:
		for i in range(N):
			for j in range(N):
				K[i][j] = exp( (-RMSD[i][j]*RMSD[i][j]) / (2*epsilonArray[i]*epsilonArray[j]) )
				D[i] += K[i][j]

				Ktilda[i][j] = K[i][j]/sqrt(D[i]*D[j])
				Dtilda[i] += Ktilda[i][j]

				P[i][j] = Ktilda[i][j]/Dtilda[i]

	return P