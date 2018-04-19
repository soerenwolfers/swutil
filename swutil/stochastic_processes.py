import numpy as np
from swutil.np_tools import integral, toeplitz_multiplication, MCSlicer
from swutil.time import Timer
import scipy.special
import scipy.linalg
from swutil.plots import plot_convergence

def logGBM(times,r,sigma,S0,d,M):
    '''
    Returns M Euler-Maruyama sample paths of log(S), where 
        dS = r*S*dt+sigma*S*dW
        S(0)=S0
    using N time steps
    
    :rtype: M x N array
    '''
    N=len(times)
    p0 = np.log(S0)#/(K*c/np.linalg.norm(c)**2)), (1, d))
    return p0 + (r-sigma**2/2)*np.tile(times,(1,d))+np.concatenate((np.zeros((M,1,d)),np.cumsum(np.sqrt(sigma**2*(times[1:]-times[:-1]))*np.random.normal(size=(M,N-1,d)),axis=1)),axis=1)

def fBrown(H,T,N,M,dW = None,cholesky = False):
    '''
    Sample fractional Brownian motion with differentiability index H 
    on interval [0,T] (H=1/2 corresponds to standard Brownian motion)
    
    :param H: Differentiability, larger than 0
    :param T: Final time
    :param N: Number of time steps
    :param M: Number of samples
    :param dW: Driving noise, optional
    '''
    alpha = 0.5-H
    times = np.linspace(0, T, N)
    dt = T/(N-1)
    if cholesky:
        if dW is not None:
            raise ValueError('Cannot use provided dW if Cholesky method is used')
        times = times[1:]
        tdt = times/np.reshape(times,(-1,1))
        tdt[np.tril_indices(N-1,-1)]=0
        cov = np.reshape(times,(-1,1))**(1-2*alpha)*(1/(1-alpha))*(tdt-1)**(-alpha)*scipy.special.hyp2f1(alpha,1-alpha,2-alpha,1/(1-tdt))
        cov[0,:] = 0
        np.fill_diagonal(cov,times**(1-2*alpha)/(1-2*alpha))
        cov[np.tril_indices(N-1,-1)] = cov.T[np.tril_indices(N-1,-1)]
        L = scipy.linalg.cholesky(cov)
        return np.concatenate((np.zeros((1,M)),L.T@np.random.normal(size=(N-1,M))))
    if dW is None:
        dW = np.sqrt(dt)*np.random.normal(size=(N-1,M))
    if H == 0.5:
        return integral(dF = dW,cumulative = True)  
    a = 1/dt/(1-alpha)*((T-times[N-2::-1])**(1-alpha)-(T-times[:0:-1])**(1-alpha))#a is array that is convolved with dW. Values arise from conditioning integral pieces on dW 
    out = toeplitz_multiplication(a,np.zeros_like(a),dW[::-1])[::-1]
    out -=a[0]*dW#Redo last bit of defining integral with exact simulation below
    cov = np.array([[ dt**(1-2*alpha)/(1-2*alpha),dt**(1-alpha)/(1-alpha)],[dt**(1-alpha)/(1-alpha),dt]])
    var = cov[0,0]-cov[0,1]**2/cov[1,1]
    out += cov[0,1]/cov[1,1]*dW #Conditional mean
    out += np.sqrt(var)*np.random.normal(size = (N-1,M))#Conditional variance
    out = np.concatenate((np.zeros((1,M)),out))
    return out

def rBergomi(H,T,eta,xi,rho,S0,r,N,M,dW=None,dW_orth=None,cholesky = False):
    times = np.linspace(0, T, N)
    dt = T/(N-1)
    times = np.reshape(times,(-1,1))
    if dW is None:
        dW = np.sqrt(dt)*np.random.normal(size=(N-1,M))
    if dW_orth is None:
        dW_orth = np.sqrt(dt)*np.random.normal(size=(N-1,M))
    dZ = rho*dW+np.sqrt(1-rho**2)*dW_orth
    Y = eta*np.sqrt(2*H)*fBrown(H,T,N,M,dW =dW,cholesky = cholesky)
    v = xi*np.exp(Y-0.5*(eta**2)*times**(2*H))
    S = S0*np.exp(integral(np.sqrt(v),dF = dZ,axis=0,cumulative = True)+integral(r - 0.5*v,F = times,axis=0,trapez=False,cumulative = True))
    return S
    
if __name__=='__main__':
    L = 8
    M = int(1e6)
    out = np.zeros((L,2))
    #N= 2**L+1
    #dt = 1/(N-1)
    #dW=np.sqrt(dt)*np.random.normal(size=(N-1,1))
    #dW_orth = np.sqrt(dt)*np.random.normal(size=(N-1,1))
    #W = np.concatenate(([[0]],np.cumsum(dW,axis = 0)))
    #W_orth = np.concatenate(([[0]],np.cumsum(dW_orth,axis = 0)))
    for l in range(1,L):
        with Timer():
            out[l] = MCSlicer(lambda M: np.exp(-0.05)*np.maximum(1-rBergomi(H=0.07,T=1,eta=1.9,xi = 0.235**2,rho = -0.9,S0=1,r=0.05,N= 2**l+1,M=M)[-1],0),M)
        print(out.__repr__())
    plot_convergence(2**np.arange(L),out)
    from matplotlib import pyplot as plt
    plt.show()
