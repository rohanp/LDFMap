
import numpy as np
cimport numpy as np

cdef extern from "rmsd.h":
	double rmsd(int n, double* x, double* y)

def calcRMSD(filename, num_atoms, num_models):
	with open(filename, 'r') as crd_file:
		coords = np.array(map(float, crd_file.read().split()))
		coords = np.reshape(coords, (num_models, num_atoms * 3))

	return np.asarray(_calcRMSD(coords, num_atoms, num_models))

cdef double[:,:] _calcRMSD(double[:,:] coords, long num_atoms, long num_models):

	cdef long i, j

	RMSD = np.zeros((num_models, num_models))
	cdef double[:,:] RMSD_view = RMSD

	for i in range(1, num_models):
		for j in range(i+1, num_models):
			RMSD[i][j] = rmsd(num_atoms, &coords[i,0], &coords[j,0])
			RMSD[j][i] = RMSD[i][j]

	return RMSD





