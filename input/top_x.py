import numpy as np
import sys

if len(sys.argv) != 4:
    print("Usage: python %s <pdb file> <scores file> <num models to keep>"%sys.argv[0])
    quit()

infile = open(sys.argv[1], 'r')
outfile = open(sys.argv[3] + '_' + sys.argv[1], 'w')
x = int(sys.argv[3])
top_x_scores = set(np.argsort(open(sys.argv[2], 'r').read().splitlines())[:x])

for line in infile:
    if len(line) == 15: #Model lines are 15 chars long
        model_num = int(line.split()[1])
        if model_num in top_x_scores:
            write = True
        else:
            write = False
    if write:
        outfile.write(line)



