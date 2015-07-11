#cython: wraparound=False, boundscheck=True, cdivision=True
#cython: profile=True, nonecheck=True, overflowcheck=True
#cython: cdivision_warnings=True, unraisable_tracebacks=True

""" A Python/Cython implementation of the Locally-Scaled Diffusion Map 
	Dimensionality Reduction Technique. Debug Version contains runtime exceptions,
	none/overflow/bounds checks, more print statements, never releases the GIL or
	uses parallelism. Running the debug version is reccomended before running the 
	faster yet less cautious normal version.
"""
__author__ = "Rohan Pandit"

import numpy as np
cimport numpy as np
from time import time
from libc.math cimport sqrt, exp

np.set_printoptions(precision=3, threshold=2000, suppress=True)

cdef extern from "rmsd.h":
	double rmsd(int n, double* x, double* y)

def main(filename, num_atoms, num_models):
	start = time()

	t0 = time()
	print("Parsing file...")
	coords = PDBParser(filename, num_atoms, num_models)
	print("File Parsed in {0} seconds".format(round(time()-start,3)))
	
	t0 = time()
	print("Calculating RMSD")
	RMSD = calcRMSD(coords, num_atoms, num_models)
	print("Calculated RMSD in {0} seconds".format(round(time()-start,3)))
	print("Saving RMSD to 'output/RSMD.txt'")
	np.savetxt('output/RMSD.txt', RMSD, fmt='%8.3f')

	t0 = time()
	print("Calculating epsilons")
	epsilons = calcEpsilons(RMSD)
	print("Calculated epsilons in {0} seconds".format(round(time()-start,3)))	
	print("Saving epsilons to output/epsilons.txt")
	np.savetxt('output/epsilons.txt', epsilons)

	print(epsilons)

	t0 = time()
	P = calcMarkovMatrix(RMSD, epsilons)
	print(P)
	print("Completed transition matrix in {0} seconds".format(round(time()-t0,3)))
	print("Saving output to output/markov.txt")
	np.savetxt('output/markov.txt', P)

	print("Done! Total time: {0} seconds".format(round(time() - start, 3)))

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

	coords = np.array(map(float, coord_list))
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

def calcRMSD(coords, num_atoms, num_models):
	""" Takes coordinates from PDB parser and calculates pairwise least 
		root-mean-squared distance between all models with given coordinates.
		Returns MxM RMSD matrix.   
	"""

	return _calcRMSD(coords, num_atoms, num_models)

cdef _calcRMSD(double[:,:] coords, long num_atoms, long num_models):
	cdef:
		long i, j
		double[:,:] RMSD_view 

	RMSD = np.zeros((num_models, num_models))
	RMSD_view = RMSD

	#for i in prange(num_models, nogil=True, schedule='dynamic', chunksize=200, num_threads=4):
	for i in range(num_models):
		if i % 10 == 0:
			print("on RMSD row {0}".format(i))
		for j in range(i+1, num_models):
			# '&' because rmsd is a C++ function that takes pointers
			RMSD_view[i][j] = rmsd(num_atoms*3, &coords[i,0], &coords[j,0])
			RMSD_view[j][i] = RMSD_view[i][j]

	return RMSD

def calcEpsilons(RMSD, cutoff = 0.03):
	""" Takes RMSD matrix and optional cutoff parameter and implements the 
		algorithm described in Clementi et al. to estimate the distance around
		each model which can be considered locally flat. Returns an array of 
		length M of these distances.
	"""

	max_epsilon = np.max(RMSD)
	print("Max RMSD: {0}".format(max_epsilon))
	possible_epsilons = np.array([(3./7.)*max_epsilon, (1./2.)*max_epsilon, (4./7.)*max_epsilon])
	epsilons = np.ones(RMSD.shape[0])

	print("Possible Epsilons: {0}".format(possible_epsilons))

	for xi in range(RMSD.shape[0]):
		print("On epsilon {0}".format(xi))
		epsilons[xi] = _calcEpsilon(xi, RMSD, possible_epsilons, cutoff)

	return epsilons

cdef double _calcEpsilon(int xi, RMSD, double[:] possible_epsilons, float cutoff) except? 1:
	cdef:
		int i, j, dim
		long a
		double[:,:] eigenvals_view
		long[:,:] status_vectors_view
		long[:] local_dim_view

	print("----- calculating eigenvalues")
	eigenvals_view = _calcMDS(xi, RMSD, possible_epsilons)

	if eigenvals_view[0][0] == -1:
		return possible_epsilons[1]

	print("----- calculating status vectors")
	status_vectors_view = _calcStatusVectors( np.asarray(eigenvals_view) )

	#if there was error in _calcStatusVectors, returns -1
	if status_vectors_view[0][0] == -1:
		return  possible_epsilons[1]

	local_dim_view = np.zeros(status_vectors_view.shape[0], dtype=long) # len = 3

	print("----- calculating local intrinsic dimensionality")
	for e in range(status_vectors_view.shape[0]):
		local_dim_view[e] = _calcIntrinsicDim(status_vectors_view[e,:])

	if xi % 10 == 0:
		f = open("Output/epsilon_{0}_data".format(xi), 'w')
		f.write("Epsilon {0} \n".format(xi))
		f.write("Eigenvalues: \n {0} \n\n".format( np.array_str(np.asarray(eigenvals_view))) )
		f.write("Status Vectors: \n {0} \n\n".format( np.array_str(np.asarray(status_vectors_view))) )
		f.write("Local Dim: \n {0} \n".format( np.array_str(np.asarray(local_dim_view))) )

	print("----- calculating epsilon")
	for dim in range(local_dim_view[e], eigenvals_view.shape[1]):
		for e in range(eigenvals_view.shape[0]):
			for i in range(dim, eigenvals_view.shape[1]):
				if cutoff < _derivative(eigenvals_view[:,i], possible_epsilons, e):
					break
			else:
				return possible_epsilons[e]

	raise Exception("Did not reach convergence. Try increasing cutoff")

cpdef double[:,:] _calcMDS(int xi, RMSD, double[:] possible_epsilons):
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
			eigenvals_view[i][j] = A[j]*A[j]

	return eigenvals_view[:,:max_neighbors]


cdef long _calcIntrinsicDim(long[:] sv) except? 1: #sv = status vector
	#TODO: This method can made more efficient by remoiving redundent checking.
	#		Would be tricky and involve lots of if statements, probably not worth.

	cdef long i

	# * 1 1 0 0 0 * in status vectors marks the separation between noise and non-noise
	for i in range(2, sv.shape[0] - 3):
		if sv[i-2] and sv[i-1] and not sv[i] and not sv[i+1] and not sv[i+2] and not sv[i+3]:
			return i

	# If last method did not find separation, the condition for separation must be more lenient
	# * 1 0 0 0 * marks separation
	for i in range(1, sv.shape[0] - 2):
		if sv[i-1] and not sv[i] and not sv[i+1] and not sv[i+2]:
			return i


	# If last method did not find separation, the condition for separation must be more lenient
	# * 1 0 0 * marks separation
	for i in range(1, sv.shape[0] - 1):
		if sv[i-1] and not sv[i] and not sv[i+1]:
			return i

	print( np.asarray(sv) )
	raise Exception(""" 
		No noise non-noise separation. The non-debug version would
		return the smallest possible epsilon in this case. However,
		if this exception is pervasive throughout your dataset,
		you may want to try defining the non-noise noise separation
		more conservatively or adjust the requirements for a status
		vector to have an entry of '1' or '0'.
					""".replace('\t',''))

cpdef long[:,:] _calcStatusVectors(eigenvals):
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
		raise Exception("""
			Status Vector fewer than 5 elements. The non-debug version would
			return the middle epsilon in this case. Try increasing the possible
			epsilon values to avoid this exception. Or maybe your dataset is just
			too small.
						""".replace('\t',''))

	#Each discrete status vector entry is set to 1 if its status vector entry is greater 
	#than twice of each of the next five status vector entries, else stays 0.
	for e in range( sv_view.shape[0] ):
		for i in range( sv_view.shape[1] - 5 ):
			if sv_view[e][i] > svx2_view[e][i+1] and sv_view[e][i] > svx2_view[e][i+2] \
			and sv_view[e][i] > svx2_view[e][i+3] and sv_view[e][i] > svx2_view[e][i+4]:
				dsv[e][i] = 1

	return dsv

cdef inline double _derivative(double[:] eigenvals, double[:] epsilons, long e):
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

	for i in range(N):
		for j in range(N):
			Ktilda[i][j] = K[i][j]/sqrt(D[i]*D[j])
			Dtilda[i] += Ktilda[i][j]

	for i in range(N):
		for j in range(N):
			P[i][j] = Ktilda[i][j]/Dtilda[i]

	np.savetxt('output/K.txt', K, fmt='%8.3f')
	np.savetxt('output/D.txt', D, fmt='%8.3f')
	np.savetxt('output/Ktilda.txt', Ktilda, fmt='%8.3f')
	np.savetxt('output/Dtilda.txt', Dtilda, fmt='%8.3f')
	np.savetxt('output/P.txt', K, fmt='%8.3f')

	print(np.sum(P))

	if np.isnan(np.sum(P)): #check if any NaNs in P
		raise Exception("NaNs in the markov transition matrix.")

	return P

