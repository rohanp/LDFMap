import numpy as np
import sys
import os 

if len(sys.argv) != 3:
    print("Usage: python %s <pdb name> <num models to keep>"%sys.argv[0])
    quit()

os.chdir(sys.argv[1])

name = sys.argv[1]
num_models = sys.argv[2]
x = int(num_models)
outfile = open(num_models + '_' + name + '.pdb', 'w')
write = False

"""
#native structure is first model in output PDB
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
"""

scores = open(name + '.scores', 'r').read().splitlines()
keep = np.sort(np.random.choice(50501, x, replace=False)) 
print(keep)
np.savetxt(num_models + '_' + name + ".kept.txt", keep, fmt='%i')

with open(name + '.pdb', 'r') as infile:
    for line in infile:
        if len(line) == 15: #Model lines are 15 chars long
            model_num = int(line.split()[1])
            if model_num in keep:
                write = True
            else:
                write = False
        if write:
            outfile.write(line)

outfile.close()