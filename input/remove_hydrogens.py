import sys

infile = open(sys.argv[1], 'r')
outfile = open(sys.argv[1] + "_modified", 'w')

for line in infile:
	if line.split()[-1] != 'H':
		outfile.write(line)
