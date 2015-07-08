import LDFMap_debug

LDFMap_debug.main('Input/short.pdb', 75, 25)


def PDBParser(filename, num_atoms, num_models):
	""" Takes PDB filename with M models and A atoms, returns Mx3A matrix
		containing the XYZ coordinates of all atoms.
	"""

	f = open(filename, 'r')
	modelnum = 0
	coord_list = []

	for line in f:
		if 'END MODEL' in line:
			modelnum += 1
		elif 33 < len(line):
			coord_list.extend(line[33:56].split())
			#writes out just the coordinates 

	coords = np.array(map(float, coord_list))
	print(coords.shape)
	coords = np.reshape(coords, (num_models, num_atoms * 3))

	return coords