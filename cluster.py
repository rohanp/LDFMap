import sys
from itertools import cycle
from time import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import k as k_b
from sklearn import cluster
from sklearn.neighbors import BallTree
from sklearn.utils import extmath
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

filename = sys.argv[1]

def main():
	pdb_name = filename.split("_")[0]

	projections = np.load("output/%s/projections.npy" % filename)[:, :2]
	RMSDs = np.load("output/%s/RMSD.npy" % filename)
	epsilons = np.load("output/%s/epsilons.npy" % filename)
	eigenvals = np.load("output/%s/eigenvals.npy" % filename)

	print("Starting mean-shift for ", filename)
	t0 = time()

	cluster_centers, labels = modified_mean_shift(RMSDs, epsilons, RMSDs)

	np.save("output/%s/cluster_centers"%filename, cluster_centers)
	np.save("output/%s/labels"%filename, labels)

	num_clusters = len(np.unique(labels))
	print("num clusters: ", num_clusters)
	#cluster_plot(projections, labels)

	print("time elapsed: %s"%(time() - t0))

	#mean_shift(RMSDs, bandwidth=epsilons)

def modified_mean_shift(X, bandwidth_array, seeds, max_iterations=300):
	"""Perform mean shift clustering of data using a gaussian kernel.

	Parameters
	----------
	X : array-like, shape=[n_samples, n_features]
		Input data.

	bandwidth : array, shape=[n_samples]
		Kernel bandwidth.

	seeds : array, shape=[n_seeds, n_features]
		Point used as initial kernel locations.

	max_iter : int, default 300
		Maximum number of iterations, per seed point before the clustering
		operation terminates (for that seed point), if has not converged yet.

	Returns
	-------
	cluster_centers : array, shape=[n_clusters, n_features]
		Coordinates of cluster centers.

	labels : array, shape=[n_samples]
		Cluster labels for each point.

	Notes
	-----
	Code adapted from scikit-learn library.

	"""
	n_points, n_features = X.shape
	stop_thresh = 1e-3 * np.mean(bandwidth_array)  # when mean has converged
	center_intensity_dict = {}
	cluster_centers = []
	ball_tree = BallTree(X)  # to efficiently look up nearby points

	def gaussian_kernel(x, points, bandwidth):
		distances = euclidean_distances(points, x)
		weights = np.exp(-1 * (distances ** 2 / bandwidth ** 2))
		return np.sum(points * weights, axis=0) / np.sum(weights)

	# For each seed, climb gradient until convergence or max_iterations                                                                                                     
	for i, weighted_mean in enumerate(seeds):
		completed_iterations = 0
		while True:
			points_within = X[ball_tree.query_radius([weighted_mean], bandwidth_array[i])[0]]
			old_mean = weighted_mean  # save the old mean                                                                                                                  
			weighted_mean = gaussian_kernel(old_mean, points_within, bandwidth_array[i])
			converged = extmath.norm(weighted_mean - old_mean) < stop_thresh

			if converged or completed_iterations == max_iterations:
				if completed_iterations == max_iterations:
					print("reached max iterations")
				cluster_centers.append(weighted_mean)
				center_intensity_dict[tuple(weighted_mean)] = len(points_within)
				break
				 
			completed_iterations += 1

	# POST PROCESSING: remove near duplicate points
	# If the distance between two kernels is less than the bandwidth,
	# then we have to remove one because it is a duplicate. Remove the
	# one with fewer points.
	sorted_by_intensity = sorted(center_intensity_dict.items(),
								 key=lambda tup: tup[1], reverse=True)
	sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
	unique = np.ones(len(sorted_centers), dtype=np.bool)
	ball_tree = BallTree(sorted_centers)

	for i, center in enumerate(sorted_centers):
		if unique[i]:
			neighbor_idxs = ball_tree.query_radius([center], np.mean(bandwidth_array))[0]
			unique[neighbor_idxs] = 0
			unique[i] = 1  # leave the current point as unique
	cluster_centers = sorted_centers[unique]

	# ASSIGN LABELS: a point belongs to the cluster that it is closest to
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cluster_centers)
	labels = np.zeros(n_points, dtype=np.int)
	distances, idxs = nbrs.kneighbors(X)
	labels = idxs.flatten()

	return cluster_centers, labels

def k_means(data, num_clusters):
	k = cluster.KMeans(n_clusters=num_clusters)
	k.fit(data)

	return k.labels_

def calcProb(energy, temp=300):
	return np.exp(-energy / (k_b * temp))

def calcEntropy(num_states, probs):
	"""
		Notes
		-----
			Derived using stirling approximation of formal 
			definition of entropy.
			..math -K_b N \sum_{k=1}^{s}p_k \ln p_k
	"""
	return -k_b * num_states * np.sum(probs * np.log(probs))

def cluster_plot(data, labels):
	colors = cycle('bgrcmyk')
	num_clusters = len(np.unique(labels))
	print("num clusters: ", num_clusters)

	for i in range(num_clusters):
		to_plot = data[ np.where(labels == i) ]
		x_plot = to_plot[:, 0]
		y_plot = to_plot[:, 1]
		plt.scatter(x_plot, y_plot, c=next(colors))

	plot(data)

def plot(data):
	x = data[:, 0]
	y = data[:, 1]
	plt.axis([min(x), max(x),min(y),max(y)])
	plt.xlabel("DC1")
	plt.ylabel("DC2")
	plt.grid()
	plt.show()
	plt.savefig("%s_plot.png"%filename, transparent=True, 
				bbox_inches='tight', figsize=(3,3), dpi=300)

if __name__ == "__main__":
	main()