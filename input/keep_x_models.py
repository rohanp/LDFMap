import sys

f = open(sys.argv[1], 'r')
x = sys.argv[2]
out = open( x + "_" + sys.argv[1], 'w')

for line in f:
	if len(line) == 15 and x in line:
		break
	out.write(line)
