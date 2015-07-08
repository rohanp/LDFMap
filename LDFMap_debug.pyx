#cython: wraparound=True, boundscheck=True, cdivision=True
#cython: profile=True, nonecheck=True, overflowcheck=True
#cython: cdivision_warnings=True, unraisable_tracebacks=True

""" An Python/Cython implementation of the Locally-Scaled Diffusion Map 
	Dimensionality Reduction Technique. Debug Version contains more prints,
	none/overflow/bounds checks, and never releases the GIL. Running the debug
	version is reccomended before running the normal version.
"""
__author__ = "Rohan Pandit"

import numpy as np
cimport numpy as np
from time import time
from libc.math cimport sqrt, exp

cdef extern from "rmsd.h":
	double rmsd(int n, double* x, double* y)

def main(filename, num_atoms, num_models):
	t0 = time()
	coords = PDBParser(filename, num_atoms, num_models)
	RMSD = calcRMSD(coords, num_atoms, num_models)
	print(RMSD)
	epsilons = calcEpsilons(RMSD)
	print(epsilons)
	P = calcMarkovMatrix(RMSD, epsilons)
	print(P)
	print("Done! {0} seconds".format(round(time() - t0, 3)))

def PDBParser(filename, num_atoms, num_models):
	""" Takes PDB filename with M models and A atoms, returns Mx3A matrix
		containing the XYZ coordinates of all atoms.
	"""

	f = open(filename, 'r')
	modelnum = 0
	coord_list = []

	for line in f:
		if 'END MODEL' in line:
			modelnum += 1
		elif 33 < len(line):
			coord_list.extend(line[33:56].split())
			#writes out just the coordinates 

	coords = np.array(map(float, coord_list))
	coords = np.reshape(coords, (num_models, num_atoms * 3))

	return coords

def calcRMSD(coords, num_atoms, num_models):
	""" Takes coordinates from PDB parser and calculates pairwise least 
		root-mean-squared distance between all models with given coordinates.
		Returns MxM RMSD matrix.   
	"""

	print("Calculating RMSD")
	return _calcRMSD(coords, num_atoms, num_models)

cdef _calcRMSD(double[:,:] coords, long num_atoms, long num_models):
	cdef:
		long i, j
		double[:,:] RMSD_view 

	RMSD = np.zeros((num_models, num_models))
	RMSD_view = RMSD

	for i in range(num_models):
		print("on RMSD row {0}".format(i))
		for j in range(i+1, num_models):
			# '&' because rmsd is a C++ function that takes pointers
			RMSD_view[i][j] = rmsd(num_atoms, &coords[i,0], &coords[j,0])
			RMSD_view[j][i] = RMSD_view[i][j]

	return RMSD

def calcEpsilons(RMSD, cutoff = 0.03):
	""" Takes RMSD matrix and optional cutoff parameter and implements the algorithm
		described in Clementi et al. to estimate the distance around each model 
		which can be considered locally flat. Returns an array of these distances,
		one for each model.
	"""

	max_epsilon = np.max(RMSD)
	possible_epsilons = np.array([(3./7.)*max_epsilon, (1./2.)*max_epsilon, (4./7.)*max_epsilon])
	epsilons = np.ones(RMSD.shape[0])

	print("Possible Epsilons: {0}".format(possible_epsilons))

	print("Calculating Epsilons")
	for xi in range(RMSD.shape[0]):
		print("On epsilon {0}".format(xi))
		epsilons[xi] = _calcEpsilon(xi, RMSD, possible_epsilons, cutoff)

	return epsilons

cdef double _calcEpsilon(int xi, RMSD, double[:] possible_epsilons, float cutoff) except? 1:
	cdef:
		int i, j
		long a
		double[:,:] eiegenvals, eigenvals_view
		long[:,:] status_vectors
		long[:] local_dim

	print("--- calculating eigenvalues")
	eigenvals_view = calcMDS(xi, RMSD, possible_epsilons)

	print("--- calculating status vectors")
	status_vectors = calcStatusVectors( np.asarray(eigenvals_view) )

	local_dim = np.zeros(status_vectors.shape[0], dtype=long)

	print("--- calculating local intrinsic dimensionality")
	for e in range(status_vectors.shape[0]):
		local_dim[e] = calcIntrinsicDim(status_vectors[e,:])

	print("--- calculating epsilon")
	for dim in range(local_dim[e], eigenvals_view.shape[1]):
		for e in range(eigenvals_view.shape[0]):
			for i in range(dim, eigenvals_view.shape[1]):
				if cutoff < derivative(eigenvals_view[:,i], possible_epsilons, e):
					break
			else:
				return possible_epsilons[e]

	raise Exception("Did not converge — returning 1")

cdef double[:,:] calcMDS(int xi, RMSD, double[:] possible_epsilons):
	cdef:
		double[:] A
		double[:,:] neighbors_matrix
		double[:,:] eigenvals_view = np.zeros( (possible_epsilons.shape[0], RMSD.shape[1]) ) 
		int i, j
		int max_neighbors = 0

	for i, e in enumerate(possible_epsilons):
		#find indexes of all neighbors
		neighbors_idxs = np.where( RMSD[xi,:] <= e )[0]
		#create RMSD matrix of just these neighbors
		neighbors_matrix = RMSD[ neighbors_idxs, : ][ :, neighbors_idxs ]
		
		if max_neighbors < neighbors_idxs.shape[0]:
			max_neighbors = neighbors_idxs.shape[0] 

		A = np.linalg.svd( neighbors_matrix, compute_uv=False )

		for j in range(A.shape[0]):
			eigenvals_view[i][j] = A[j]*A[j]

	return eigenvals_view[:,:max_neighbors]


cdef long calcIntrinsicDim(long[:] sv) except? 1: #sv = status vector
	cdef long i

	# * 1 1 0 0 0 * in status vectors marks the separation between noise and non-noise
	for i in range(2, sv.shape[0] - 4):
		if sv[i-2] and sv[i-1] and not sv[i] and not sv[i+3] and not sv[i+4]:
			return i

	raise Exception("ERROR: No noise non-noise separation — returning 1")

cdef long[:,:] calcStatusVectors(eigenvals):
	cdef:
		double[:,:] sv_view, svx2_view
		int e, i
		cdef long[:,:] dsv

	#status vector = gap between eigenvalues
	sv = eigenvals[:, :eigenvals.shape[1] - 1] - eigenvals[:, 1:] 
	sv_view = sv
	svx2 = sv*2
	svx2_view = svx2

	try:
		dsv = np.zeros(( sv.shape[0], sv.shape[1] - 5 ), dtype=long)
	except ValueError:
		raise Exception("Status Vector fewer than 5 elements")

	#Each discrete status vector entry is set to 1 if its status vector entry is greater 
	#than twice of each of the next five status vector entries, else stays 0.
	for e in range( sv_view.shape[0] ):
		for i in range( sv_view.shape[1] - 5 ):
			if sv_view[e][i] > svx2_view[e][i+1] and sv_view[e][i] > svx2_view[e][i+2] \
			and sv_view[e][i] > svx2_view[e][i+3] and sv_view[e][i] > svx2_view[e][i+4] \
			and sv_view[e][i] > svx2_view[e][i+5]:
				dsv[e][i] = 1

	return dsv

cdef inline double derivative(double[:] eigenvals, double[:] epsilons, long e):
	cdef double derivative 
	if e == 0:
		derivative = (eigenvals[1] - eigenvals[0])/(epsilons[1] - epsilons[0])
	elif e == 2:
		derivative = (eigenvals[2] - eigenvals[1])/(epsilons[2] - epsilons[1])
	else:
		derivative = (eigenvals[2] - eigenvals[0])/(epsilons[2] - epsilons[0])

	return derivative

def calcMarkovMatrix(RMSD, epsilons):
	""" Takes the MxM RMSD matrix and the array of epsilons of length M,
		returns the MxM Markov transition matrix
	"""

	return np.asarray( _calcMarkovMatrix(RMSD, epsilons, RMSD.shape[0]) )

cdef double[:,:] _calcMarkovMatrix(double[:,:] RMSD, double[:] epsilons, int N):	
	cdef: 
		int i, j
		#all are memoryviews
		double[:] D = np.zeros(N)
		double[:] Dtilda = np.zeros(N)
		double[:,:] K = np.zeros((N,N))
		double[:,:] Ktilda = np.zeros((N,N))
		double[:,:] P = np.zeros((N,N))

	for i in range(N):
		for j in range(N):
			K[i][j] = exp( (-RMSD[i][j]*RMSD[i][j]) / (2*epsilons[i]*epsilons[j]) )
			D[i] += K[i][j]

			Ktilda[i][j] = K[i][j]/sqrt(D[i]*D[j])
			Dtilda[i] += Ktilda[i][j]

			P[i][j] = Ktilda[i][j]/Dtilda[i]

	return P

