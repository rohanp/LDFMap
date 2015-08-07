import LDFMap
import LDFMap_debug
import numpy as np
from matplotlib import pyplot
from time import time
import os

def main():
	run("input/Met-Enk_AMBER/Met-Enk_combined.pdb", 40, 6180)

def run(filename, num_atoms, num_models):
	start = time()

	t0 = time()
	print("Parsing file...")
	coords = LDFMap.PDBParser(filename, num_atoms, num_models)
	print("File Parsed in {0} seconds".format(round(time()-start,3)))

	name = filename.split('/')[-1][:-4]

	if not os.path.exists("output/" + name):
		os.makedirs("output/" + name)
	
	t0 = time()
	print("Calculating RMSD")
	RMSD = LDFMap.calcRMSDs(coords, num_atoms, num_models)
	print("Calculated RMSD in {0} seconds".format(round(time()-start,3)))
	print("Saving RMSD to 'Output/RSMD.npy'\n")
	np.save('output/{name}/RMSD.npy'.format(name=name), RMSD)

	t0 = time()
	print("Calculating epsilons")
	epsilons = LDFMap.calcEpsilons(RMSD, prints=True)
	print("Calculated epsilons in {0} seconds".format(round(time()-start,3)))	
	print("Saving epsilons to output/{name}/epsilons.npy\n".format(name=name))
	np.save('output/{name}/epsilons.npy'.format(name=name), epsilons)

	t0 = time()
	print("Calculating Markov transition matrix")
	P = LDFMap.calcMarkov(RMSD, epsilons)
	print("Completed markov transition matrix in {0} seconds".format(round(time()-t0,3)))
	print("Saving output to Output/markov.npy\n".format(name=name))
	np.save('output/{name}/markov.npy'.format(name=name), P)

	t0 = time()
	print("Calculating eigenvalues and eigenvectors")
	eigenvals, eigenvecs = LDFMap.calcEig(P)
	print("Completed eigendecomposition in {0} seconds".format(round(time()-t0, 3)))
	print("Saving output to output/{name}/eigenvals.npy".format(name=name))
	print("Saving output to output/{name}/eigenvecs.npy\n".format(name=name))
	np.save('output/{name}/eigenvals.npy'.format(name=name), eigenvals)
	np.save('output/{name}/eigenvecs.npy'.format(name=name), eigenvecs)

	t0 = time()
	print("Calculating projections and accumulated variance")
	projections = LDFMap.calcProj(P, eigenvecs)
	accumVar = LDFMap.calcAccumVar(eigenvals)
	print("Completed in {0} seconds\n\n".format(round(time()-t0, 3)))
	print("Saving projections to output/{name}/projections.npy".format(name=name))
	np.save('output/{name}/projections.npy'.format(name=name), projections)
	np.save('output/{name}/accum_var.npy'.format(name=name), accumVar)

	print("Done! Total time: {0} seconds".format(round(time() - start, 3)))


def time_trials():
	xs = [1000, 2000, 4000, 6000]
	ys = []

	for x in xs:
		t0 = time()
		print("Calculating RMSD")
		RMSD = LDFMap.calcRMSDs(coords[:x,:], 40, x)
		print("Calculated RMSD in {0} seconds".format(round(time()-t0,3)))
		np.savetxt('Output/RMSD.txt', RMSD)

		print("Calculating Epsilons")
		epsilons = LDFMap.calcEpsilons(RMSD)
		print("Calculated Epsilons in {0} seconds".format(round(time()-t0,3)))
		np.savetxt('Output/epsilons.txt', epsilons)

		print("Calculating Markov Matrix")
		P = LDFMap.calcMarkovMatrix(RMSD, epsilons)
		print("Calculated Markov Matrix in {0} seconds".format(round(time()-t0,3)))
		np.savetxt('Output/markov.txt', P)

		ys.append(time() - t0)

	print(ys)

	pyplot.plot(xs, ys)
	pyplot.xlabel("Num Models")
	pyplot.ylabel("Runtime (seconds)")
	pyplot.show()



if __name__ == "__main__": main()


