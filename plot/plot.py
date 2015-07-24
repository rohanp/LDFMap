import numpy as np
from matplotlib import pyplot
import matplotlib.patches as mpatches

def main():
	print("reading input")
	proj = np.load("output/Met-Enk_AMBER/projections.npy")
	accumVar = np.load("output/Met-Enk_AMBER/accum_var.npy")
	print("plotting")
	plotAccumVar(accumVar)
	plotProj(proj)

def plotAccumVar(accumVar):
	fig, ax=pyplot.subplots()
	n=accumVar.shape[0]
	pyplot.plot(range(n), accumVar, "ro")
	pyplot.axis([0,n,0,100])
	pyplot.hlines(80,0,n)
	pyplot.grid()
	pyplot.xlabel("Eigenvalue Number")
	pyplot.ylabel("Accumulated Variance")

def plotProj(projMatrix):    
	x=projMatrix[:,0]
	y=projMatrix[:,1]
	pyplot.plot(x, y, 'o')
	#pyplot.plot(x[:2000],y[:2000],"ro", label='values 0-20')
	#pyplot.plot(x[2000:4000],y[2000:4000], "bo", label='values 20-100')
	#pyplot.plot(x[4000:],y[4000:],"go", label='values 100-end')
	pyplot.axis([min(x), max(x),min(y),max(y)])
	pyplot.xlabel("DC1")
	pyplot.ylabel("DC2")
	pyplot.grid()
	pyplot.show()

if __name__ == "__main__":
	main()

