# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 19:29:11 2014

@author: rohan
"""
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as pyplot
from matplotlib.font_manager import FontProperties
import fileinput

if len(sys.argv) < 3:
    print ("usage: python %s <fileName> <number of points to plot> <maxEpsilon>" % str(sys.argv[0]))
    quit()

fileName=sys.argv[1]
n=int(sys.argv[2])
maxEpsilon=float(sys.argv[3])

os.chdir("..")
os.chdir("output")
os.chdir(fileName)
name="projMatrix_"+fileName+".txt"
projMatrix=np.loadtxt(name,dtype=np.double)


evals_sorted=np.loadtxt("evalsonly_"+fileName+".txt",dtype=np.double)
epsilonArray=[]

lines=fileinput.input("epsilonArray_"+fileName+".txt")
for line in lines:   
    index=line.index("=")
    epsilonArray.append(float(line[index+2:]))


accVars = evals_sorted.copy()
nevals = accVars.shape[0]

for i in range(1, nevals):
    previous = accVars[i-1]
    accVars[i] += previous

for i in range(0, nevals):
    accVars[i] /= accVars[nevals-1]
    accVars[i] *= 100.0


#n=how many evals you want to graph
fig, ax=pyplot.subplots()
evals_plot=np.zeros(n)
for i in range(n):
    evals_plot[i]=math.e**(-evals_sorted[len(evals_sorted)-1-i])
pyplot.plot(range(n), evals_plot, "ro")
pyplot.axis([0,n,0,1])
pyplot.xlabel("Eigenvalue Number")
pyplot.grid()
pyplot.ylabel("exp(-lambda)")
pyplot.savefig("EigenValuesPlot_"+fileName +".png", transparent='True', bbox_inches='tight', figsize=(3,3), dpi=300)


fig, ax=pyplot.subplots()
pyplot.plot(range(n), accVars[:n], "ro")
pyplot.axis([0,n,0,100])
pyplot.hlines(80,0,n)
pyplot.grid()
pyplot.xlabel("Eigenvalue Number")
pyplot.ylabel("Accumulated Variance")
pyplot.savefig("AccumVarPlot_"+fileName+".png", transparent='True', bbox_inches='tight', figsize=(3,3), dpi=300)


x=projMatrix[:,0]
y=projMatrix[:,1]
fig, ax = pyplot.subplots()
pyplot.plot(x[:20],y[:20],'ro',label='2LWC')
pyplot.plot(x[20:100],y[20:100], 'bo', label='1PLW')
pyplot.plot(x[100:],y[100:], 'go', label='1PLX')
pyplot.axis([min(x), max(x),min(y),max(y)])
pyplot.xlabel("DC1")
pyplot.ylabel("DC2")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
fontP = FontProperties()
fontP.set_size('small')
pyplot.grid()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop= fontP, numpoints=1)
pyplot.savefig("ProjPlot_"+fileName+".png", transparent='True', bbox_inches='tight', figsize=(3,3), dpi=300)


fig, ax=pyplot.subplots()
a=(3./7.)*maxEpsilon
b= (1./2.)*maxEpsilon
c=(4./7.)*maxEpsilon
xlocs=np.arange(3)
width=.35
plotArray=[0,0,0]
for i in epsilonArray:
    if abs(i-a)<.1:
        plotArray[0]+=1
    if abs(i-b)<.1:
        plotArray[1]+=1
    if abs(i-c)<.1:
        plotArray[2]+=1
rects=pyplot.bar(xlocs+width,plotArray,width,color='b')

ax.set_ylabel('Number of Atoms')
ax.set_xlabel('Epsilon')
ax.set_xticks(xlocs+width+width/2.)
ax.set_xticklabels((str(a)[:4],str(b)[:4],str(c)[:4]))
pyplot.savefig("epsilonPlot"+fileName +".png", transparent='True', bbox_inches='tight', figsize=(4,4), dpi=300)


