import LDFMap
import numpy as np
from matplotlib import pyplot
from time import time

coords = LDFMap.PDBParser('Input/Met-Enk_AMBER.pdb', 75, 6000)

xs = [1000, 2000, 4000]
ys = []

for x in xs:
	t0 = time()
	print("Calculating RMSD")
	RMSD = LDFMap.calcRMSDs(coords[:x,:], 40, x)
	print("Calculated RMSD in {0} seconds".format(round(time()-t0,3)))

	t0 = time()
	print("Calculating Epsilons")
	epsilons = LDFMap.calcEpsilons(RMSD)
	print("Calculated Epsilons in {0} seconds".format(round(time()-t0,3)))

	t0 = time()
	print("Calculating Markov Matrix")
	P = LDFMap.calcMarkovMatrix(RMSD, epsilons)
	print("Calculated Markov Matrix in {0} seconds".format(round(time()-t0,3)))

	ys.append(time() - t0)

print(ys)

pyplot.plot(xs, ys)
pyplot.xlabel("Num Models")
pyplot.ylabel("Runtime (seconds)")
pyplot.show()



