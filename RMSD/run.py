#!/usr/bin/env python
""" Prepares input, compiles, and runs calcRMSD.pyx """
__author__ = "Rohan Pandit"

import sys
import numpy as np

import pyximport 
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import calcRMSD

def main(name, num_atoms, num_models):
	#Example usage: python run.py Met-Enk 75 180

	with open('../../input/' + name + '.crd', 'r') as crd_file:
		coords = np.array(map(float, crd_file.read().split()))
		coords = np.reshape(coords, (num_models, num_atoms * 3))

	RMSD = calcRMSD.calcRMSD(coords)


if __name__ == "__main__":
	main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))