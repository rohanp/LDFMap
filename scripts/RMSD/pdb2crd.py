import sys
import os

def main():
	#Example Usage: python3 pdb2crd.py Met-Enk 180
	name = sys.argv[1]
	numAtoms = int(sys.argv[2])

	os.chdir("..")
	os.chdir("..")
	os.chdir("input")
	f = open(name + '.pdb', 'r')
	modelnum = 0

	os.chdir("..")
	os.chdir("scripts/RMSD/data")
	out=open(name + '.crd', 'w')

	for l in f:
		if 'END MODEL' in l:
			modelnum+=1
			#out=open(name + str(modelnum) + '.crd', 'w')
		elif len(l)>33:
			out.write(l[33:56] + '\n')

	splitFile(name+'.crd', numAtoms)

def splitFile(inFileName, numAtoms):
	infile = open(inFileName, 'r')
	outfile = open(inFileName+"_1", 'w')

	modelNum=1
	count=1
	for line in infile:
		if count > numAtoms:
			modelNum += 1
			count = 1
			outfile = open(inFileName+"_"+str(modelNum),'w')
		
		if count == numAtoms: # to take care of ending with newline problem
			line = line[:-2]
		
		outfile.write(line)
		count+=1

if __name__=="__main__":
	main()
