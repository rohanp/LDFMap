import LDFMap_debug
import numpy as np

def PDBParser(filename, num_atoms, num_models):
	""" Takes PDB filename with M models and A atoms, returns Mx3A matrix
		containing the XYZ coordinates of all atoms.
	"""

	f = open(filename, 'r')
	modelnum = 0
	coord_list = []

	for line in f:
		len_ = len(line)
		if 'END' in line:
			modelnum += 1
		elif len_ == 79: 
			#columns 33 to 56 contain the xyz coordinates
			coord_list.extend(line[33:56].split())

	coords = np.array(map(float, coord_list))
	try:
		coords = np.reshape(coords, (num_models, num_atoms * 3))
	except ValueError:
		raise Exception("""
			Could not parse PDB file. Make sure your PDB file is 
			formatted like the example, 'Input/Met-Enk.pdb' and 
			that you entered the correct values for num_atoms and 
			num_models. 
						""")

	return coords

LDFMap_debug.main('Input/Met-Enk.pdb', 75, 180)

