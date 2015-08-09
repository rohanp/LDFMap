import LDFMap
import numpy as np
from time import time
import os
import sys

def main():
        if len(sys.argv) == 3:
                filename = sys.argv[1]
                num_models = int(sys.argv[2])
        else:
                filename = "input/SOD1/20000_SOD1.pdb"
                num_models = 20000

        num_atoms = get_num_atoms(filename)
        print("num atoms: ", num_atoms)

        run(filename, num_atoms, num_models)

def get_num_atoms(filename):
        num_atoms = 0
        
        with open(filename, 'r') as f:
                for line in f:
                        if 'ATOM' in line:
                                num_atoms += 1
                        if 'END' in line:
                                break
        return num_atoms

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


if __name__ == "__main__": main()


