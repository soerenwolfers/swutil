import numpy as np
from swutil.validation import NDim
from math import floor

def is_1d(array):
    return np.squeeze(array).ndim == 1
def unitv(ind,size):
    x=np.zeros(size)
    x[ind] = 1
    return x
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
