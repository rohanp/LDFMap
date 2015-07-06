# by Rohan Pandit

import sys
import os, re
import subprocess
import numpy as np
from time import time

processes = set()

def main():	
	if len(sys.argv) < 4:
		print ("usage: python %s <inFileName> <numAtoms> <numModels> <maxProcesses>" % str(sys.argv[0]))
		quit()
	
	startTime= time()
	
	inFileName = sys.argv[1]
	numAtoms = int(sys.argv[2])
	numModels = int(sys.argv[3])
	maxProcesses = int(sys.argv[4])

	os.chdir("data")

	calcRMSDParallel(inFileName, numModels, maxProcesses)
	print(time()-startTime)


def calcRMSDParallel(inFileName, numModels, maxProcesses):

	for i in range(1,numModels+1):
		s= []
		s.append("./ComputeRMSDBetweenCRDs")
		s.append(inFileName+"_"+str(i))
		s.append(inFileName)
		s.append(str(numModels))
		s.append(inFileName+"_RMSD_"+str(i))
		s.append(inFileName + "_aligned_" + str(i)+".remove")

		processes.add(subprocess.Popen(s))

		if(len(processes)>=maxProcesses):
			print("########### ON MODEL %s ############"%i) 
			os.wait()
			processes.difference_update([p for p in processes if p.poll() is not None])

	#wait for all processes to finish
	for p in processes:
		if p.poll() is None:
			p.wait()

def purge():
	files = [f for f in os.listdir('.') if os.path.isfile(f)]
	for f in files:
		if f.endswith(".remove"):
			os.remove(os.path.join('.', f))

if __name__=="__main__":
	main()

