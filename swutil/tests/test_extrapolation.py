import numpy as np
import swutil
from swutil.plots import plot_convergence
from matplotlib import pyplot
from swutil.np_tools import extrapolate
noise = 1e-3
L=30
h = 2.**(-np.arange(0,L))
w = 2.**(np.arange(0,L))
v = np.sin(h)
v = v+np.random.normal(scale = noise,size=(L,))
plot_convergence(w,v)
v2,w2 = extrapolate(v,w,degree = -1)
plot_convergence(w2,v2)
pyplot.show()