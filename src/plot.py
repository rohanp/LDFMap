from __future__ import division
from matplotlib import pyplot

labels = ['SVD Call', 'Rest of Program']
sizes = [409/438, (438-409)/438]

pyplot.pie(sizes, labels=labels, colors=('lightskyblue','yellowgreen'),
        autopct='%1.1f%%', startangle=90)

pyplot.axis('equal')

pyplot.show()
