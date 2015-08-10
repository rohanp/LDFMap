""" Evaluating the effectiveness of a variety of clustering algorithms 
"""
__author__ = "Rohan Pandit" 

import sys
from itertools import cycle
from time import time
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import k as k_b
from sklearn import cluster
from sklearn.neighbors import BallTree
from sklearn.utils import extmath
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner

filename = sys.argv[1]

K_MEANS = False
AFFINITY_PROP = False
MEAN_SHIFT = False
AGGLOMERATIVE = False
DBSCAN = True

#Example Usage: python cluster.py 5000_SOD1
def main():
	####################### Loading Files ##########################
	pdb_name = filename.split("_")[0]

	algorithms = ['k_means', 'affinity_prop', 'affinity_prop_eps', 
				  'mean_shift', 'mean_shift_eps', 'agglomerative',
				  'DBSCAN', 'DBSCAN_eps', ]
	for name in algorithms:
		if not os.path.exists("output/%s/%s"%(filename, name)):	
			os.makedirs("output/%s/%s"%(filename, name))

	projections = np.load("output/%s/projections.npy" % filename)[:, :2]
	RMSDs = np.load("output/%s/RMSD.npy" % filename)
	epsilons = np.load("output/%s/epsilons.npy" % filename)
	eigenvals = np.load("output/%s/eigenvals.npy" % filename)

	########################## K-Means ############################
	if K_MEANS:
		t0 = time()
		print("Starting K-Means for", filename)

		k = cluster.KMeans(n_clusters=num_clusters, n_jobs=-1).fit(RMSDs)

		np.save("output/%s/k_means/cluster_centers"%filename, k.cluster_centers_)
		np.save("output/%s/k_means/labels"%filename, k.labels_)

		print("num clusters: ", k.cluster_centers_.shape[0])
		print("time elapsed: %s \n"%(time() - t0))
		#cluster_plot(projections, labels)

	################### Affinity Propagation #######################
	if AFFINITY_PROP:
		t0 = time()
		print("Starting Affinity Propagation for", filename)

		af = cluster.AffinityPropagation(verbose=True, affinity='precomputed').fit(RMSDs)

		np.save("output/%s/affinity_prop/cluster_centers"%filename, af.cluster_centers_)
		np.save("output/%s/affinity_prop/labels"%filename, af.labels_)

		print("num clusters: ", af.cluster_centers_.shape[0])
		print("time elapsed: %s \n"%(time() - t0))

	################ Affinity Propagation with Epsilons ###############
	if AFFINITY_PROP:
		t0 = time()
		print("Starting Affinity Propagation with epsilons for", filename)

		af = cluster.AffinityPropagation(preference=epsilons, verbose=True, 
											affinity='precomputed').fit(RMSDs)

		np.save("output/%s/affinity_prop_eps/cluster_centers"%filename, af.cluster_centers_)
		np.save("output/%s/affinity_prop_eps/labels"%filename, af.labels_)

		print("num clusters: ", cluster_centers.shape[0])
		print("time elapsed: %s \n"%(time() - t0))

	############################ Mean Shift  #############################
	if MEAN_SHIFT:
		t0 = time()
		print("Starting Mean Shift for", filename)

		ms = cluster.MeanShift(bandwidth=np.mean(RMSD), bin_seeding=True).fit(RMSDs)

		np.save("output/%s/mean_shift/cluster_centers"%filename, ms.cluster_centers_)
		np.save("output/%s/mean_shift/labels"%filename, ms.labels_)

		print("num clusters: ", ms.cluster_centers_.shape[0])
		print("time elapsed: %s \n"%(time() - t0))

	####################### Mean Shift with Epsilons  #########################
	if MEAN_SHIFT:
		t0 = time()
		print("Starting Mean Shift with epsilons for", filename)

		cluster_centers, labels = variable_bw_mean_shift(RMSDs, bandwidth_array=epsilons)

		np.save("output/%s/mean_shift_eps/cluster_centers"%filename, cluster_centers)
		np.save("output/%s/mean_shift_eps/labels"%filename, labels)

		print("num clusters: ", cluster_centers.shape[0])
		print("num clusters: ", len(set(labels)))
		print("time elapsed: %s \n"%(time() - t0))

	##### Density-Based Spatial Clustering of Applications with Noise (DBSCAN) ####
	if DBSCAN:
		t0 = time()
		print("Starting DBSCAN for", filename)

		d = cluster.DBSCAN(eps=np.mean(RMSDs), metric='precomputed',
							algorithm='ball_tree', min_samples=3).fit(RMSDs)

		np.save("output/%s/DBSCAN/cluster_centers"%filename, d.components_)
		np.save("output/%s/DBSCAN/labels"%filename, d.labels_)

		print("num clusters: ", len(set(d.labels_)))
		print("time elapsed: %s \n"%(time() - t0))

	############################ DBSCAN with Epsilons ###########################
	if DBSCAN:
		t0 = time()
		print("Starting DBSCAN with epsilons for", filename)

		cluster_centers, labels = variable_eps_DBSCAN(RMSDs, epsilons, min_samples=2)

		np.save("output/%s/DBSCAN_eps/cluster_centers"%filename, cluster_centers)
		np.save("output/%s/DBSCAN_eps/labels"%filename, labels)

		print(d.labels_)
		print(np.unique(d.labels_).shape[0])
		print("num clusters: ", len(set(d.labels_)))
		print("time elapsed: %s \n"%(time() - t0))

########################## Clustering Algorithms #######################

def variable_bw_mean_shift(X, bandwidth_array, seeds=None, max_iterations=300):
	"""Variable bandwidth mean shift with gaussian kernel

	Parameters
	----------
	X : array-like, shape=[n_samples, n_features]
		Input data.

	bandwidth : array[float], shape=[n_samples]
		Kernel bandwidth.

	seeds : array[float, float], shape=(n_seeds, n_features), optional
		Point used as initial kernel locations. Default is
		setting each point in input data as a seed.

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

	if not seeds:
		seeds = X 

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

def variable_eps_DBSCAN(X, eps_array, min_samples=5):
	""" Density-Based Spatial Clustering of Applications with Noise

	Parameters
	----------
	X : array[float, float], shape=(n_samples,n_features)
		Similarity matrix

	eps_array : array[float], shape=(n_samples)
		The maximum distance between two points for them to be considered 
		to be in the same neighborhood, applied locally.

	Returns
	--------
	cluster_centers : array, shape=[n_clusters, n_features]
		Coordinates of cluster centers.

	labels : array, shape=[n_samples]
		Cluster labels for each point.

	Notes
	-----
	Code adapted from scikit-learn library 
	"""
	# Calculate neighborhood for all samples. This leaves the original point
	# in, which needs to be considered later (i.e. point i is in the
	# neighborhood of point i. While True, its useless information)
	neighborhoods = np.array([np.where(x <= eps_array[i])[0] for i, x in enumerate(X)])

	n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])

	# Initially, all samples are noise.
	labels = -np.ones(X.shape[0], dtype=np.intp)

	# A list of all core samples found.
	core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
	dbscan_inner(core_samples, neighborhoods, labels)

	return np.where(core_samples)[0], labels


######################### Plotting ########################

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


# def calcProb(energy, temp=300):
# 	return np.exp(-energy / (k_b * temp))

# def calcEntropy(num_states, probs):
# 	"""
# 		Notes
# 		-----
# 			Derived using stirling approximation of formal 
# 			definition of entropy.
# 			..math -K_b N \sum_{k=1}^{s}p_k \ln p_k
# 	"""
# 	return -k_b * num_states * np.sum(probs * np.log(probs))

