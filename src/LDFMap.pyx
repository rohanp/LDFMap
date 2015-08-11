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
import random

cdef extern from "rmsd.h" nogil:
	double rmsd(int n, double* x, double* y)

def PDBParser(filename, num_atoms, num_models):
	""" Parses PDB file for XYZ coordinates of all atoms in a format 
		that can be used by `calcRMSDs`.

		Parameters
		----------
		filename : str
			Path to PDB file.
		num_atoms : int
		num_models : int

		Returns
		-------
		array[float, float], shape = (num_models, 3 * num_atoms)
			Contains the XYZ coordinates of all atoms. 

		Raises
		------
		ValueError
			If the values of `num_atoms` and `num_models` are not
			consistent with the contents in `filename`. 
	"""

	f = open(filename, 'r')
	modelnum = 0
	coord_list = []

	for line in f:
		if 'END' in line:
			modelnum += 1
		elif 'ATOM' in line: 
			#columns 33 to 56 contain the xyz coordinates
			coord_list.extend(line[33:56].split())

	coords = np.array(coord_list, dtype=float)
	try:
		coords = np.reshape(coords, (num_models, num_atoms * 3))
	except ValueError:
		coordnum = coords.shape[0]
		raise Exception("""
			Could not parse PDB file. Make sure your PDB file is 
			formatted like the example, 'input/Met-Enk.pdb' and 
			that you entered the correct values for num_atoms and 
			num_models.

			Coodrinates read: {0}
			Models read: {1}
			Num atoms: {2}
		""".replace('\t','').format(coordnum, modelnum, coordnum/modelnum))

	return coords

def calcRMSDs(coords, num_atoms, num_models):
	""" Calculates pairwise least root-mean-squared distance 
		between all models.

		Parameters
		----------
		coords : array[float, float], shape = (num_models, 3 * num_atoms)
			Array of XYZ coordinates of all atoms in all models.
			Should use `PDBParser` to create this array
		num_atoms : int
		num_models : int

		Returns
		-------
		array[float, float], shape = (num_models, num_models)
			Containing pairwise lRMSD between all models. 
		
	"""

	return _calcRMSDs(coords, num_atoms, num_models)

cdef _calcRMSDs(double[:,:] coords, long num_atoms, long num_models):
	cdef:
		long i, j
		double[:,:] RMSD_view 

	RMSD = np.zeros((num_models, num_models))
	RMSD_view = RMSD

	with nogil:
		for i in range(num_models):
			for j in range(i+1, num_models):
				# '&' because rmsd is a C++ function that takes pointers
				RMSD_view[i, j] = rmsd(num_atoms*3, &coords[i,0], &coords[j,0])
				RMSD_view[j, i] = RMSD_view[i, j]

	return RMSD
	
def calcEpsilons(RMSDs, cutoff = 0.03, sample_size=None):
	""" Implements algorithm described in Clementi et al. to estimate the 
		distance around each model which can be considered locally flat.

		Parameters
		----------
		RMSDs : array[float, float]
			Containes pairwise leat root-mean-squared distances between
			all models.
		cutoff : float, optional
			See Clementi et al.

		Returns
		-------
		array[float]
			Contains the epsilon value for each model, the distance 
			around each model which can be considered locally flat.
			shape = (`RMSD.shape[0]`,)

		Notes
		------
		Squashes exceptions, uses print statements instead due to
		common occurance of non-fatal exceptions.

		References
		----------
		.. [1]  Rohrdanz M, Zheng W, Maggioni M, Clementi C (2011) 
				Determination of reaction coordinates via locally 
				scaled diffusion map. J Chem Phys 134: 124116.

	"""

	print("Max RMSD: {0}".format(np.max(RMSDs))) 

	if not sample_size: sample_size = RMSDs.shape[0]

	epsilons = np.ones(RMSDs.shape[0])

	for xi in range(RMSDs.shape[0]):
		if not xi % 10: print("On Epsilon {0}".format(xi))
		epsilons[xi] = _calcEpsilon(xi, RMSDs, cutoff, sample_size)

	return epsilons

cdef double _calcEpsilon(int xi, RMSD, float cutoff, int sample_size):
	cdef:
		int i, j, dim
		double[:,:] eigenvals
		long[:,:] status_vectors
		long[:] local_dim
		double[:,:] noise_eigenvals
		double[:] possible_epsilons

	max_epsilon = np.max(RMSD[xi])
	possible_epsilons = np.array([3./7., 1./2., 4./7.]) * max_epsilon
	RMSD_sample = np.zeros((sample_size, sample_size))

	if sample_size != RMSD.shape[0]:
		sample_idxs = [xi] + random.sample(range(RMSD.shape[0]), sample_size - 1)
		RMSD_sample = RMSD[sample_idxs, :][:, sample_idxs]
	else:
		RMSD_sample = RMSD

	eigenvals = _calcMDS(xi, RMSD_sample, possible_epsilons)

	#if there was error in _calcMDS, it returns -1
	if eigenvals[0, 0] == -1:
		print("Error in _calcMDS!")
		return possible_epsilons[1]

	status_vectors = _calcStatusVectors( np.asarray(eigenvals) )

	#if there was error in _calcStatusVectors, it returns -1
	if status_vectors[0, 0] == -1:
		print("Error in _calcStatusVectors!")
		return  possible_epsilons[1]

	local_dim = np.zeros(status_vectors.shape[0], dtype=long) # len = 3

	with nogil:
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

	raise Exception("ERROR: Did not reach convergence on epsilon {0}.".format(xi))

cdef double[:,:] _calcMDS(int xi, RMSD, double[:] possible_epsilons):
	cdef:
		double[:] A
		double[:,:] neighbors_matrix
		double[:,:] eigenvals_view = np.zeros((possible_epsilons.shape[0], RMSD.shape[1])) 
		int i, j
		int max_neighbors = 0
		double eps

	for i, eps in enumerate(possible_epsilons):
		#find indexes of all neighbors. First row of RMSD matrix is model xi.
		neighbors_idxs = np.where( RMSD[0,:] <= eps )[0]

		#ERROR: no neighbors. insert a -1 so that calcEpsilon knows
		#an error has occured and will return the middle epsilon.
		if neighbors_idxs.shape[0] < 2:
			return np.zeros( (10, 10) ) - 1.

		#create RMSD matrix of just these neighbors
		neighbors_matrix = RMSD[ neighbors_idxs, : ][ :, neighbors_idxs ]
		
		#to find out what shape the eigenvalue array should be
		if max_neighbors < neighbors_idxs.shape[0]:
			max_neighbors = neighbors_idxs.shape[0] 

		A = np.linalg.svd( neighbors_matrix, compute_uv=False )

		for j in range(A.shape[0]):
			eigenvals_view[i, j] = A[j]*A[j]

	return eigenvals_view[:,:max_neighbors]


cdef long _calcIntrinsicDim(long[:] sv) nogil: #sv = status vector
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

def calcMarkov(RMSDs, epsilons):
	""" Calculates the Markov transition matrix, which contains the 
		transitional probability between two states. Uses algorithm
		from Clementi et al.

		Parameters
		----------
		RMSDs : array[float, float]
			Contains pairwise least root-mean-squared distances between
			each model.
		epsilons : array[float]
			Contains the distances around each model that can be 
			considered locally flat.

		Returns 
		-------
		array[float, float]
			Contains transitional probability between two given states.

	"""

	return np.asarray( _calcMarkovMatrix(RMSDs, epsilons, RMSDs.shape[0]) )

cdef double[:,:] _calcMarkovMatrix(double[:,:] RMSD, double[:] epsilons, int N):	
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
				K[i, j] = exp( (-RMSD[i, j] * RMSD[i, j]) / (2*epsilons[i] * epsilons[j]) )
				D[i] += K[i, j]

		for i in range(N):
			for j in range(N):
				Ktilda[i, j] = K[i, j] / sqrt(D[i]*D[j])
				Dtilda[i] += Ktilda[i, j]

		for i in range(N):
			for j in range(N):
				P[i, j] = Ktilda[i, j] / Dtilda[i]

	return P
 
def calcEig(P):
	""" Eigendecomposes the transition matrix and then sorts eigenvalues and
		eigenvectors in decreasing order by eigenvalue, returns sorted versions.

		Parameters
		----------
		P : array[float, float]
			Markov matrix, contains transitional probability between models.

		Returns
		-------
		eigenvals : array[float]
			Contains eigenvalues sorted in descending order.
		eigenvalues : array[float, float]
			Contains eigenvectors (in columns) corresponding
			to each eigenvalue.

	""" 
	eigenvals, eigenvecs = np.linalg.eig(P)

	#sort eigenvals/vecs
	idxs = np.argsort(eigenvals)[::-1]
	eigenvals = eigenvals[ idxs ]
	eigenvecs = eigenvecs[:, idxs]

	return eigenvals, eigenvecs


def calcProj(P, eigenvecs):
	""" Projects the transition matrix on to the eigenvectors, returns projection.

		Parameters
		----------
		P : array[float, float]
			Markov matrix, contains transitional probability between models.
		eigenvecs : array[float, float]
			Contains eigenvectors (in columns).

		Returns
		-------
		array[float, float]
			Projection of transition matrix onto eigenvectors.
			.. math:: PE_v^t 

		Notes
		-----
		Eigenvectors MUST be in columns.

	"""
	return np.dot(P, eigenvecs.T)

def calcAccumVar(eigenvals):
	""" Calculates accumulated variance captured by eigenvalues

		Parameters
		----------
		eigenvals : array[float]
			Contains eigenvalues sorted in descending order.

		Returns
		-------
		array[float]
			Accumulated variance captured by eigenvalues.
	"""
	#only positive eigenvals (returns copy)
	accVars = eigenvals[ 0 < eigenvals ]

	accVars = np.cumsum(accVars)
	accVars /= accVars[ accVars.shape[0] - 1 ]
	accVars *= 100

	return accVars


