import numpy as np
import sys
import os 

if len(sys.argv) != 3:
    print("Usage: python %s <pdb name> <num models to keep>"%sys.argv[0])
    quit()

name = sys.argv[1]
os.chdir(name)

infile = open(name + '.pdb', 'r')
scores_file = open(name + '.scores', 'r')
x = int(sys.argv[2])

outfile = open('top_' + sys.argv[2] + '_' + name + '.pdb', 'w')
top_x_scores = set(np.argsort(scores_file.read().splitlines())[:x])

write = False
#write native structure as first model in output PDB
with open(name + '.exp.pdb', 'r') as expfile:
    for line in expfile:
        items = line.split()

        if "MODEL" in line and len(items) == 2:
            if '1' in line:
                write = True
            else:
                break

        if "TER" in line and write:
            break

        if items[-1] == 'H':
            continue

        if write:
            outfile.write(line[:70].strip() + '\n')

    outfile.write("END MODEL\n")

write = False
#write rest of structures
for line in infile:
    #if MODEL line
    if 'MODEL' in line and not 'END' in line: 
        model_num = int(line.split()[1])

        if model_num in top_x_scores:
            write = True
        else:
            write = False

    if write:
        outfile.write(line)



