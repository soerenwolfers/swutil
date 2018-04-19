import numpy as np
from swutil.plots import plot_convergence
from swutil.np_tools import extrapolate
L=100
K=50
base=0.2
coeff = np.random.rand(1,L)
hs= 2.**(-np.arange(1,K))
w=2**np.arange(1,K)
hs = np.reshape(hs,(-1,1))
hf= hs**(base*np.arange(1,L+1))
T = coeff*hf
values = np.sum(T,axis=1)
from matplotlib import pyplot as plt
plot_convergence(w,values,reference=0)
plot_convergence(w[0:],extrapolate(values,degree=-1,base=base/2),reference=0)
plt.show()

