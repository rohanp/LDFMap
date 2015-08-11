import numpy as np
from matplotlib import pyplot
import matplotlib.patches as mpatches
import sys
from matplotlib import cm
import os

#usage: python plot/plot.py Met-Enk_AMBER
name = sys.argv[1]
protein_name = sys.argv[1].split('_')[1]

def main():

	if not os.path.exists("output/%s/plots"%name):	
			os.makedirs("output/%s/plots"%name)

	print("reading input")
	scores = np.loadtxt("input/%s/%s.scores"%(protein_name, protein_name))
	models = np.loadtxt("input/%s/%s.kept.txt"%(protein_name, name), dtype=int)

	proj = np.load("output/%s/projections.npy"%name)
	accumVar = np.load("output/%s/accum_var.npy"%name)
	RMSD = np.load("output/%s/RMSD.npy"%name)

	print("plotting")
	plotAccumVar(accumVar)
	plotProjEnergy(proj, scores, models)
	plotProjRMSD(proj, RMSD, models)

def plotAccumVar(accumVar):
	fig, ax=pyplot.subplots()
	n = accumVar.shape[0]
	pyplot.plot(range(n), accumVar, "ro")
	pyplot.axis([0,n,0,100])
	pyplot.hlines(80,0,n)
	pyplot.grid()
	pyplot.xlabel("Eigenvalue Number")
	pyplot.ylabel("Accumulated Variance")
	pyplot.savefig("output/%s/plots/accum_var.png"%name, transparent=True, 
				bbox_inches='tight', figsize=(3,3), dpi=300)
	pyplot.show()


def plotProjEnergy(projMatrix, scores, models):
	x = projMatrix[:,0]
	y = projMatrix[:,1]

	scores = np.sort(scores[ models ])
	lowest_energy = scores[0]
	highest_energy = scores[-1]
	num_models = models.shape[0]

	print(lowest_energy, highest_energy, num_models)
	colors = np.linspace(lowest_energy, highest_energy, num_models)

	pyplot.scatter(x[1:], y[1:], c=colors, cmap=cm.jet)
	pyplot.plot(x[0], y[0], 'kx', mew=3)[0].set_ms(10)

	bar = pyplot.colorbar()
	bar.set_label("Energy (low to high)")
	#pyplot.plot(x[:2000],y[:2000],"ro", label='values 0-20')
	#pyplot.plot(x[2000:4000],y[2000:4000], "bo", label='values 20-100')
	#pyplot.plot(x[4000:],y[4000:],"go", label='values 100-end')
	pyplot.axis([min(x), max(x),min(y),max(y)])
	pyplot.xlabel("DC1")
	pyplot.ylabel("DC2")
	pyplot.grid()
	pyplot.savefig("output/%s/plots/proj_energy.png"%name, transparent=True, 
					bbox_inches='tight', figsize=(3,3), dpi=300)
	pyplot.show()

def plotProjRMSD(projMatrix, RMSD, models):
	x = projMatrix[:,0]
	y = projMatrix[:,1]

	print(np.max(RMSD[0,1:]))
	print(np.min(RMSD[0,1:]))

	close = np.where( RMSD[0] < 5 )
	medium = np.where( np.logical_and( RMSD[0] < 10, 5 < RMSD[0] ) )
	far = np.where( 10 < RMSD[0] )

	pyplot.scatter(x[far], y[far], c='k', label='10> A')
	pyplot.scatter(x[medium], y[medium], c='lightblue', label='5-10 A')
	pyplot.scatter(x[close], y[close], c='orange', label='5< A')

	#pyplot.plot(x[0], y[0], 'rx', mew=2, label='native')[0].set_ms(8)

	pyplot.axis([min(x), max(x),min(y),max(y)])
	pyplot.xlabel("DC1")
	pyplot.ylabel("DC2")
	pyplot.grid()
	pyplot.legend(loc='upper right', scatterpoints=1, numpoints=1)

	pyplot.savefig("output/%s/plots/proj_rmsd.png"%name, transparent=True, 
					bbox_inches='tight', figsize=(3,3), dpi=300)
	pyplot.show()


def plotMetEnk(projMatrix):
	x = projMatrix[:,0]
	y = projMatrix[:,1]

	x2 = projMatrix[180:,0]
	y2 = projMatrix[180:,1]

	pyplot.scatter(x2[:2000], y2[:2000], c='darkgreen')
	pyplot.scatter(x2[2000:4000], y2[2000:4000], c='darkred')
	pyplot.scatter(x2[4000:], y2[4000:], c='darkblue')

	pyplot.scatter(x[:60], y[:60], c='lime')
	pyplot.scatter(x[60:120], y[60:120], c='r')
	pyplot.scatter(x[120:180], y[120:180], c='b')

	pyplot.axis([min(x), max(x),min(y),max(y)])
	pyplot.xlabel("DC1")
	pyplot.ylabel("DC2")
	pyplot.grid()

	pyplot.show()	

if __name__ == "__main__":
	main()

