import numpy as np
from swutil.validation import NDim
from math import floor
from scipy import ifft, fft
from swutil.aux import split_integer

def is_1d(array):
    return np.squeeze(array).ndim == 1

def unitv(ind,size):
    x=np.zeros(size)
    x[ind] = 1
    return x

def one_changed(a,i,v):
    t = np.array(a)
    t[i] = v
    return t
    
def MCSlicer(f,M,bucket = int(1e4),length = None):
    Ms = split_integer(M,bucket = bucket,length = length)  
    slices = [(np.mean(y,axis = 0),np.mean(y**2,axis=0)) for y in map(f,Ms)]
    mean,sm = np.average(slices,axis=0,weights = Ms)
    return mean,np.sqrt(sm-mean**2)/np.sqrt(M)

def extrapolate(x, w = None, degree = None,base = 1):
    x = np.array(x)
    if degree in (None, -1):
        full = True
        out = np.zeros_like(x)
        out[0] = x[0]
        if w is not None:
            w = np.cumsum(w)
        degree = len(x)-1
    else:
        full = False
    for i in range(degree):
        x = x[:-1]+2**(base*(i+1))/(2**(base*(i+1))-1)*(x[1:]-x[:-1])
        if full:
            out[i+1] = x[0]
        elif w is not None:
            w = w[:-1]+w[1:]
    if not full:
        out=x
    if w is not None:
        return out,w
    else:
        return out

def integral(A=None,dF=None,F=None,axis = 0,trapez = False,cumulative = False):
    '''
    Turns an array A of length N (the function values in N points)
    and an array dF of length N-1 (the masses of the N-1 intervals)
    into an array of length N (the integral \int A dF at N points, with first entry 0)
    
    :param A: Integrand (optional, default ones, length N)
    :param dF: Integrator (optional, default ones, length N-1)
    :param F: Alternative to dF (optional, length N)
    :param trapez: Use trapezoidal rule (else left point)
    '''
    ndim = max(v.ndim for v in (A,dF,F) if v is not None)
    def broadcast(x):
        new_shape = [1]*ndim
        new_shape[axis] = -1
        return np.reshape(x,new_shape)
    if F is not None:
        assert(dF is None)
        if F.ndim<ndim:
            F = broadcast(F)
        N = F.shape[axis]
        dF = F.take(indices = range(1,N),axis = axis)-F.take(indices = range(N-1),axis = axis)
    elif dF is not None:
        if dF.ndim<ndim:
            dF = broadcast(dF)
        N = dF.shape[axis]+1
    else:
        if A.ndim<ndim:
            A = broadcast(A)
        N = A.shape[axis]
    if A is not None:
        if trapez:
            midA = (A.take(indices = range(1,N),axis = axis)+A.take(indices = range(N-1),axis = axis))/2
        else:
            midA = A.take(indices=range(N-1),axis=axis)
        if dF is not None:
            dY = midA*dF
        else:
            dY = midA
    else:
        dY = dF
    pad_shape = list(dY.shape)
    pad_shape[axis] = 1
    pad = np.zeros(pad_shape)
    if cumulative:
        return np.concatenate((pad,np.cumsum(dY,axis = axis)),axis = axis)
    else:
        return np.sum(dY,axis = axis)
    
def toeplitz_multiplication(a,b,v):
    '''
    Multiply Toeplitz matrix with first row a and first column b with vector v
    
    Normal matrix multiplication would require storage and runtime O(n^2);
    embedding into a circulant matrix and using FFT yields O(log(n)n)
    '''
    a = np.reshape(a,(-1))
    b = np.reshape(b,(-1))
    n = len(a)
    c = np.concatenate((a[[0]],b[1:],np.zeros(1),a[-1:0:-1]))
    p = ifft(fft(c)*fft(v.T,n=2*n)).T#fft autopads input with zeros if n is supplied
    if np.all(np.isreal(a)) and np.all(np.isreal(b)) and np.all(np.isreal(v)):
        return np.real(p[:n])
    else:
        return p[:n]
 
def grid_evaluation(X, Y, f,vectorized=True):
    '''
    Evaluate function on given grid and return values in grid format
    
    Assume X and Y are 2-dimensional arrays containing x and y coordinates, 
    respectively, of a two-dimensional grid, and f is a function that takes
    1-d arrays with two entries. This function evaluates f on the grid points
    described by X and Y and returns another 2-dimensional array of the shape 
    of X and Y that contains the values of f.
    :param X: 2-dimensional array of x-coordinates
    :param Y: 2-dimensional array of y-coordinates
    :param f: function to be evaluated on grid
    :param vectorized: `f` can handle arrays of inputs
    :return: 2-dimensional array of values of f
    '''
    XX = np.reshape(np.concatenate([X[..., None], Y[..., None]], axis=2), (X.size, 2), order='C')
    if vectorized:
        ZZ = f(XX)
    else:
        ZZ = np.array([f(x) for x in XX])
    return np.reshape(ZZ, X.shape, order='C')  
 
def precision_round(x, precision=0):
    return round(x, precision - int(floor(np.log10(abs(x))))) 

def orthonormal_complement_basis(v:NDim(1)):
    '''
    Return orthonormal basis of complement of vector.
    
    :param v: 1-dimensional numpy array 
    :return: Matrix whose .dot() computes coefficients w.r.t. an orthonormal basis of the complement of v 
        (i.e. whose row vectors form an orthonormal basis of the complement of v)
    '''
    _, _, V = np.linalg.svd(np.array([v]))
    return V[1:]

def weighted_median(values, weights):
    '''
    Returns element such that sum of weights below and above are (roughly) equal
    
    :param values: Values whose median is sought
    :type values: List of reals
    :param weights: Weights of each value
    :type weights: List of positive reals
    :return: value of weighted median
    :rtype: Real
    '''
    if len(values) == 1:
        return values[0]
    if len(values) == 0:
        raise ValueError('Cannot take median of empty list')
    values = [float(value) for value in values]
    indices_sorted = np.argsort(values)
    values = [values[ind] for ind in indices_sorted]
    weights = [weights[ind] for ind in indices_sorted]
    total_weight = sum(weights)
    below_weight = 0
    i = -1
    while below_weight < total_weight / 2:
        i += 1
        below_weight += weights[i]
    return values[i]
