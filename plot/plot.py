import numpy as np
from matplotlib import pyplot
import matplotlib.patches as mpatches
import sys
from matplotlib import cm

#usage: python plot/plot.py Met-Enk_AMBER
name = sys.argv[1]
num_models = int(sys.argv[2])

def main():
	print("reading input")
	proj = np.load("output/%s/projections.npy"%name)
	#accumVar = np.loadtxt("output/%s/accum_var.txt"%name)
	scores = np.loadtxt("input/2FS1.scores")

	print("plotting")
	#plotAccumVar(accumVar)
	plotProj(proj, scores)

def plotAccumVar(accumVar):
	fig, ax=pyplot.subplots()
	n = accumVar.shape[0]
	pyplot.plot(range(n), accumVar, "ro")
	pyplot.axis([0,n,0,100])
	pyplot.hlines(80,0,n)
	pyplot.grid()
	pyplot.xlabel("Eigenvalue Number")
	pyplot.ylabel("Accumulated Variance")
	#pyplot.show()

def plotProj(projMatrix, scores):
	x = projMatrix[:,0]
	y = projMatrix[:,1]

	lowest_energy = np.sort(scores)[0]
	highest_energy = np.sort(scores)[num_models]
	colors = np.linspace(lowest_energy, highest_energy, num_models)

	pyplot.scatter(x, y, c=colors, cmap=cm.Blues)
	bar = pyplot.colorbar()
	bar.set_label("Energy (low to high)")
	#pyplot.plot(x[:2000],y[:2000],"ro", label='values 0-20')
	#pyplot.plot(x[2000:4000],y[2000:4000], "bo", label='values 20-100')
	#pyplot.plot(x[4000:],y[4000:],"go", label='values 100-end')
	pyplot.axis([min(x), max(x),min(y),max(y)])
	pyplot.xlabel("DC1")
	pyplot.ylabel("DC2")
	pyplot.grid()
	pyplot.savefig("output/%s/projections.png"%name, transparent=True, 
					bbox_inches='tight', figsize=(3,3), dpi=300)
	pyplot.show()



if __name__ == "__main__":
	main()

