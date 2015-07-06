#cython: wraparound=False, boundscheck=False, cdivision=True
#cython: profile=False, nonecheck=False
#filename: calcEpsilon.pyx
#by Rohan Pandit

import numpy as np
cimport numpy as np
from cython.parallel import prange

cpdef double calcEpsilon(int xi, RMSD, double[:] possibleEpsilons, float cutoff):
	cdef:
		int i, j
		long a
		double[:,:] eiegenvals, eigenvals_view
		long[:,:] status_vectors
		long[:] local_dim

	print("evals")
	eigenvals_view = calcMDS(xi, RMSD, possibleEpsilons)

	print("status_vectors")
	status_vectors = calcStatusVectors( np.asarray(eigenvals_view) )

	local_dim = np.zeros(status_vectors.shape[0], dtype=long)

	print("calcIntrinsicDim")
	for i in range(status_vectors.shape[0]):
		local_dim[i] = calcIntrinsicDim(status_vectors[i,:])
		print(i)

	#do derivative thingies 
	return local_dim[2]

cdef double[:,:] calcMDS(int xi, RMSD, double[:] possibleEpsilon):
	cdef:
		double[:] A
		double[:,:] neighborsMatrix
		double[:,:] eigenvals_view = np.zeros( (possibleEpsilon.shape[0], RMSD.shape[1]) ) 
		int i, j
		int max_neighbors = 0

	for i, e in enumerate(possibleEpsilon):
		#find indexes of all neighbors
		neighbors_idxs = np.where( RMSD[xi,:] <= e )[0]
		#create RMSD matrix of just these neighbors
		neighborsMatrix = RMSD[ neighbors_idxs, : ][ :, neighbors_idxs ]
		
		if max_neighbors < neighbors_idxs.shape[0]:
			max_neighbors = neighbors_idxs.shape[0] 

		A = np.linalg.svd( neighborsMatrix, compute_uv=False )

		for j in range(A.shape[0]):
			eigenvals_view[i][j] = A[j]*A[j]

	return eigenvals_view[:,:max_neighbors]


cdef long calcIntrinsicDim(long[:] sv): #sv = status vector
	cdef long i

	try:
		for i in range(2, sv.shape[0] - 4):
			if sv[i-2] and sv[i-1] and not sv[i] and not sv[i+3] and not sv[i+4]:
				return i
	except Exception as e:
		raise e

	print("No noise non-noise separation")
	return 5
	"""
	with gil:
		raise Exception("No noise non-noise separation")
	"""

cdef long[:,:] calcStatusVectors(eigenvals):
	cdef:
		double[:,:] sv_view, svx2_view
		int e, i
		cdef long[:,:] dsv

	sv = eigenvals[:, :eigenvals.shape[1] - 1] - eigenvals[:, 1:] 
	sv_view = sv
	svx2 = sv*2
	svx2_view = svx2

	try:
		dsv = np.zeros(( sv.shape[0], sv.shape[1] - 5 ), dtype=long) #dsv = discrete status vector
	except ValueError:
		raise Exception("Status Vector fewer than 5 elements")

	for e in range( sv_view.shape[0] ):
		for i in range( sv_view.shape[1] ):
			if sv_view[e][i] > svx2_view[e][i+1] and sv_view[e][i] > svx2_view[e][i+2] \
			and sv_view[e][i] > svx2_view[e][i+3] and sv_view[e][i] > svx2_view[e][i+4] \
			and sv_view[e][i] > svx2_view[e][i+5]:
				dsv[e][i] = 1

	return dsv

cdef inline int derivativeUnderCutoff(double[:] eigenvals, double[:] epsilons, float cutoff):
	cdef float derivative = (eigenvals[1] - eigenvals[0])/(epsilons[1] - epsilons[0])
	return derivative < cutoff

