__author__ = 'arenduchintala'
import numpy as np
import random

fname = '../../lowres_features/trained-full-cov-no-overlap/sirs/sirs-cond-deviation-feat-6-states-20-iter-mean.txt'
means = np.loadtxt(fname)  # open(fname, 'r').readlines()
means = np.delete(means, 0, 1)
import matplotlib.pyplot as plt

colors = "bgrmyk"
symbols = "o*sd^+"
base = range(len(means[0]))
for m in range(means.shape[0]):

    plt.plot(base, means[m], marker=symbols[m % len(symbols)], c=colors[m % len(colors)], label=str(m))

plt.ylabel('HR Deviation')
plt.xlabel('Sample')
plt.title('Mean Value of States')
#plt.show()
plt.savefig('sirs-6-states-20-iterations.png')

'''

        b: blue
        g: green
        r: red
        c: cyan
        m: magenta
        y: yellow
        k: black
        w: white

'''