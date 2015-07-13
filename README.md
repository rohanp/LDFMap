# What is this?
A fast, Cython implementation of the Locally Scaled Diffusion Map
dimensionality reduction technique intended for use in computational biology.

# Installation
Simply download this repo, move "LDFMap.so" to your project directory and `import LDFMap`. 

The debug version has type/overflow checks, raises exceptions, has print statements showing progress, can be profiled, (all at the cost of speed), and is reccomended if the main program is giving errors. Only Python 2.x is currently supported.

# Public Methods
```cython
def PDBParser(filename, num_atoms, num_models):
    """ Takes PDB filename with M models and A atoms, returns Mx3A matrix
	    containing the XYZ coordinates of all atoms.
	"""

def calcRMSD(coords, num_atoms, num_models):
    """ Takes coordinates from PDB parser and calculates pairwise least 
    	root-mean-squared distance between all models with given coordinates.
    	Returns MxM RMSD matrix.   
    """

def calcEpsilons(RMSD, cutoff = 0.03):
    """ Takes RMSD matrix and optional cutoff parameter and implements the 
    	algorithm described in Clementi et al. to estimate the distance around
    	each model which can be considered locally flat. Returns an array of 
    	length M of these distances.
    """
def calcMarkovMatrix(RMSD, epsilons):
	""" Takes the MxM RMSD matrix and the array of epsilons of length M,
		returns the MxM Markov transition matrix, which gives the transitional
		probability between any two structures.
	"""
```

# Example Usage:
```cython
import LDFMap

coords = LDFMap.PDBParser("Met-Enk.pdb", num_atoms = 75, num_models = 180) 
#arguments can be keyworded or positional

RMSDs = LDFMap.calcRMSDs(coords, 75, 180)

epsilons = calcEpsilons(RMSD)

P = calcMarkovMatrix(RMSDs, epsilons)
```

# Dev Information
You can modify the program to fit your needs. After making changes to `src/LDFMap.pyx`, these changes must
be compiled by `setup.py`, which must be modified for your system (see comments in setup.py for more information). Run `python2.7 setup.py build_ext --inplace` to compile the Cython script into C extension that you may then import into Python and use.

Compiling requires a number of dependencies:

* 	Cython
* 	NumPy
* 	LAPACK
* 	BLAS

# TODO
Singular Value Decomposition (using NumPy's linear algebra package) has been a huge bottleneck. Currently trying to find a way to get around this.

