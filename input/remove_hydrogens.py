import sys

infile = open(sys.argv[1], 'r')
outfile = open('modified_'+sys.argv[1], 'w')

for line in infile:
	if len(line) == 79:
		if line[77] == 'H':
			continue
	outfile.write(line)
