by Rohan Pandit

import numpy as np

def calcEpsilon(xi, RMSD, possibleEpsilons, cutoff):

	print("Calculating eigenvalues")
	eigenvals_view = calcMDS(xi, RMSD, possibleEpsilons)

	print("Calculating status vectors")
	status_vectors = calcStatusVectors( np.asarray(eigenvals_view) )

	local_dim = np.zeros(status_vectors.shape[0], dtype=long)

	print("Calculating local intrinsic dimensionality")
	for e in range(status_vectors.shape[0]):
		local_dim[e] = calcIntrinsicDim(status_vectors[i,:])

	print("Calculating epsilon")
	for dim in range(local_dim[e], eigenvals_view.shape[1]):
		for e in range(eigenvals_view.shape[0]):
			for i in range(dim, eigenvals_view.shape[1]):
				if cutoff < derivative(eigenvals_view[:,i], possibleEpsilons, e):
					break
			else:
				return possibleEpsilons[e]

	print("Finished calculations, returning %s"%local_dim[2])
	return local_dim[2]

def calcMDS(xi, RMSD, possibleEpsilons):

	max_neighbors = 0

	for i, e in enumerate(possibleEpsilons):
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


def calcIntrinsicDim(sv): #sv = status vector

	# * 1 1 0 0 0 * in status vectors marks the separation between noise and non-noise
	for i in range(2, sv.shape[0] - 4):
		if sv[i-2] and sv[i-1] and not sv[i] and not sv[i+3] and not sv[i+4]:
			return i

	print("No noise non-noise separation — returning 1")
	return 1

def calcStatusVectors(eigenvals):
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

def derivative(eigenvals, epsilons, e):
	if e == 0:
		derivative = (eigenvals[1] - eigenvals[0])/(epsilons[1] - epsilons[0])
	elif e == 2:
		derivative = (eigenvals[2] - eigenvals[1])/(epsilons[2] - epsilons[1])
	else:
		derivative = (eigenvals[2] - eigenvals[0])/(epsilons[2] - epsilons[0])

	return derivative

def main(numModels):

	r = np.random.rand(numModels, numModels)
	RMSD = (np.tril(r) + np.tril(r).T) * (1 - np.eye(r.shape[0]))*4

	maxEpsilon = np.max(RMSD)
	possibleEpsilons = np.array([(3./7.)*maxEpsilon, (1./2.)*maxEpsilon, (4./7.)*maxEpsilon])
	epsilons = []

	for xi in range(numModels):
		print("######## epsilon {0} ########".format(xi))
		epsilons.append(calcEpsilon(xi, RMSD, possibleEpsilons, 0.3))

	print(epsilons)

if __name__ == "__main__": main(100)