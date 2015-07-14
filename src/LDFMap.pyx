#cython: wraparound=False, boundscheck=False, cdivision=True
#cython: profile=False, nonecheck=False, overflowcheck=False
#cython: cdivision_warnings=False, unraisable_tracebacks=False

""" A Python/Cython implementation of the Locally-Scaled Diffusion Map 
	Dimensionality Reduction Technique. 
"""
__author__ = "Rohan Pandit"

import numpy as np
cimport numpy as np
from time import time
from cython.parallel import prange
from libc.math cimport sqrt, exp

cdef extern from "rmsd.h" nogil:
	double rmsd(int n, double* x, double* y)

def PDBParser(filename, num_atoms, num_models):
	""" Takes PDB filename with M models and A atoms, returns Mx3A matrix
		containing the XYZ coordinates of all atoms.
	"""

	f = open(filename, 'r')
	modelnum = 0
	coord_list = []

	for line in f:
		len_ = len(line)
		if 'END' in line:
			modelnum += 1
		elif len_ == 79: 
			#columns 33 to 56 contain the xyz coordinates
			coord_list.extend(line[33:56].split())

	coords = np.array( list(map(float, coord_list)) )
	try:
		coords = np.reshape(coords, (num_models, num_atoms * 3))
	except ValueError:
		raise Exception("""
			Could not parse PDB file. Make sure your PDB file is 
			formatted like the example, 'Input/Met-Enk.pdb' and 
			that you entered the correct values for num_atoms and 
			num_models. 
						""".replace('\t',''))

	return coords

def calcRMSDs(coords, num_atoms, num_models):
	""" Takes coordinates from PDB parser and calculates pairwise least 
		root-mean-squared distance between all models with given coordinates.
		Returns MxM RMSD matrix.   
	"""

	return _calcRMSDs(coords, num_atoms, num_models)

cdef _calcRMSDs(double[:,:] coords, long num_atoms, long num_models):
	cdef:
		long i, j
		double[:,:] RMSD_view 

	RMSD = np.zeros((num_models, num_models))
	RMSD_view = RMSD

	for i in range(num_models):
		for j in range(i+1, num_models):
			# '&' because rmsd is a C++ function that takes pointers
			RMSD_view[i, j] = rmsd(num_atoms*3, &coords[i,0], &coords[j,0])
			RMSD_view[j, i] = RMSD_view[i, j]

	return RMSD

def calcEpsilons(RMSD, cutoff = 0.03):
	""" Takes RMSD matrix and optional cutoff parameter and implements the 
		algorithm described in Clementi et al. to estimate the distance around
		each model which can be considered locally flat. Returns an array of 
		length M of these distances.
	"""

	print("Max RMSD: {0}".format(np.max(RMSD)))
	epsilons = np.ones(RMSD.shape[0])

	for xi in range(RMSD.shape[0]):
		print("On epsilon {0}".format(xi))
		epsilons[xi] = _calcEpsilon(xi, RMSD, cutoff)

	return epsilons

cdef double _calcEpsilon(int xi, RMSD, float cutoff) except? 1:
	cdef:
		int i, j, dim
		double[:,:] eigenvals
		long[:,:] status_vectors
		long[:] local_dim
		double[:,:] noise_eigenvals
		double[:] possible_epsilons

	max_epsilon = np.max(RMSD[xi])
	possible_epsilons = np.array([(3./7.)*max_epsilon, (1./2.)*max_epsilon, (4./7.)*max_epsilon])

	eigenvals = _calcMDS(xi, RMSD, possible_epsilons)

	#if there was error in _calcStatusVectors, it returns -1
	if eigenvals[0, 0] == -1:
		return possible_epsilons[1]

	status_vectors = _calcStatusVectors( np.asarray(eigenvals) )

	#if there was error in _calcStatusVectors, it returns -1
	if status_vectors[0, 0] == -1:
		return  possible_epsilons[1]

	local_dim = np.zeros(status_vectors.shape[0], dtype=long) # len = 3

	for e in range(status_vectors.shape[0]):
		local_dim[e] = _calcIntrinsicDim(status_vectors[e,:])


	noise_eigenvals = np.zeros((eigenvals.shape[0], eigenvals.shape[1] - np.min(local_dim)))

	with nogil:

		for e in range(noise_eigenvals.shape[0]):
			for i in range(noise_eigenvals.shape[1]):
				if local_dim[e] <= i:
					noise_eigenvals[e, i - local_dim[e]] = eigenvals[e,i]  

		for dim in range(noise_eigenvals.shape[1]):
			for e in range(noise_eigenvals.shape[0]):
				for i in range(dim, noise_eigenvals.shape[1]):
					if cutoff < _derivative(noise_eigenvals[:,i], possible_epsilons, e):
						break
				else:
					return possible_epsilons[e]

	raise Exception("ERROR: Did not reach convergence. Try increasing cutoff")

cdef double[:,:] _calcMDS(int xi, RMSD, double[:] possible_epsilons):
	cdef:
		double[:] A
		double[:,:] neighbors_matrix
		double[:,:] eigenvals_view = np.zeros( (possible_epsilons.shape[0], RMSD.shape[1]) ) 
		int i, j
		int max_neighbors = 0

	for i, e in enumerate(possible_epsilons):
		#find indexes of all neighbors
		neighbors_idxs = np.where( RMSD[xi,:] <= e )[0]

		#ERROR: no neighbors. insert a -1 so that calcEpsilon knows
		#an error has occured and will return the middle epsilon.
		if neighbors_idxs.shape[0] < 2:
			return np.zeros( (10, 10) ) - 1.

		#create RMSD matrix of just these neighbors
		neighbors_matrix = RMSD[ neighbors_idxs, : ][ :, neighbors_idxs ]
		
		if max_neighbors < neighbors_idxs.shape[0]:
			max_neighbors = neighbors_idxs.shape[0] 

		A = np.linalg.svd( neighbors_matrix, compute_uv=False )

		for j in range(A.shape[0]):
			eigenvals_view[i, j] = A[j]*A[j]

	return eigenvals_view[:,:max_neighbors]


cdef long _calcIntrinsicDim(long[:] sv): #sv = status vector
	cdef long i

	with nogil:
		# * 1 1 0 0 0 * in status vectors marks the separation between noise and non-noise
		for i in range(2, sv.shape[0] - 3):
			if sv[i-2] and sv[i-1] and not sv[i] and not sv[i+1] and not sv[i+2] and not sv[i+3]:
				return i

		# If last method did not find separation, the condition for separation must be more lenient
		# * 1 0 0 0 * marks separation
		for i in range(1, sv.shape[0] - 2):
			if sv[i-1] and not sv[i] and not sv[i+1] and not sv[i+2]:
				return i

	print("No noise-non noise separation. Returning local dim = 1.")
	return 1

cdef long[:,:] _calcStatusVectors(eigenvals):
	cdef:
		double[:,:] sv_view, svx2_view
		int e, i
		cdef long[:,:] dsv

	#status vector = gap between eigenvalues
	sv = eigenvals[:, :eigenvals.shape[1] - 1] - eigenvals[:, 1:] 
	sv_view = sv
	svx2 = sv * 2
	svx2_view = svx2

	try:
		dsv = np.zeros(( sv.shape[0], sv.shape[1] - 5 ), dtype=long)
	except ValueError:
		print("ERROR: Status Vector fewer than 5 elements. Returning 1/2 * max epsilon")
		return np.zeros(sv.shape, dtype=long) - 1

	#Each discrete status vector entry is set to 1 if its status vector entry is greater 
	#than twice of each of the next five status vector entries, else stays 0.
	with nogil:
		for e in range( sv_view.shape[0] ):
			for i in range( sv_view.shape[1] - 5 ):
				if sv_view[e, i] > svx2_view[e, i+1] and sv_view[e, i] > svx2_view[e, i+2] \
				and sv_view[e, i] > svx2_view[e, i+3] and sv_view[e, i] > svx2_view[e, i+4]:
					dsv[e, i] = 1

	return dsv

cdef inline double _derivative(double[:] eigenvals, double[:] epsilons, long e) nogil:
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

	with nogil:
		for i in range(N):
			for j in range(N):
				K[i, j] = exp( (-RMSD[i, j]*RMSD[i, j]) / (2*epsilons[i]*epsilons[j]) )
				D[i] += K[i, j]

		for i in range(N):
			for j in range(N):
				Ktilda[i, j] = K[i, j]/sqrt(D[i]*D[j])
				Dtilda[i] += Ktilda[i, j]

		for i in range(N):
			for j in range(N):
				P[i, j] = Ktilda[i, j]/Dtilda[i]

	return P

def main(filename, num_atoms, num_models):
	start = time()

	t0 = time()
	print("Parsing file...")
	coords = PDBParser(filename, num_atoms, num_models)
	print("File Parsed in {0} seconds".format(round(time()-start,3)))
	
	t0 = time()
	print("Calculating RMSD")
	RMSD = calcRMSDs(coords, num_atoms, num_models)
	print("Calculated RMSD in {0} seconds".format(round(time()-start,3)))
	print("Saving RMSD to 'Output/RSMD.txt'")
	np.savetxt('Output/RMSD.txt', RMSD, fmt='%8.3f')

	t0 = time()
	print("Calculating epsilons")
	epsilons = calcEpsilons(RMSD)
	print("Calculated epsilons in {0} seconds".format(round(time()-start,3)))	
	print("Saving epsilons to Output/epsilons.txt")
	np.savetxt('Output/epsilons.txt', epsilons)

	t0 = time()
	P = calcMarkovMatrix(RMSD, epsilons)
	print("Completed transition matrix in {0} seconds".format(round(time()-t0,3)))
	print("Saving output to Output/markov.txt")
	np.savetxt('Output/markov.txt', P)

	print("Done! Total time: {0} seconds".format(round(time() - start, 3)))
