'''
Various plotting functions
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patches
from matplotlib.colors import LightSource
import matplotlib
import warnings
import matplotlib2tikz
from matplotlib.pyplot import savefig
from swutil.np_tools import weighted_median, grid_evaluation
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D  # @UnresolvedImport. Axes3D is needed for projection = '3d' below @UnusedImport
from swutil.validation import Float, Dict, List, Tuple, Bool, String, Integer
from numpy import meshgrid
from matplotlib.backends.backend_pdf import PdfPages
import os
from collections import OrderedDict

def _classify(properties,into='path'):
    '''
    Turns keyword pairs into path
    '''
    subdirs = []
    for property,value in properties.items():  # @ReservedAssignment
        if Bool.valid(value):
            subdirs.append(('' if value else 'not_')+'{}'.format(property))
        elif String.valid(value):
            subdirs.append(value)
        elif (Float|Integer).valid(value):
            subdirs.append('{}{}'.format(property,value))
        else:
            subdirs.append('{}_{}'.format(property,value))
    path = os.path.join(*subdirs)
    head,_ = os.path.split(path)
    if not os.path.exists(head):
        os.makedirs(head)
    return path

def save(*name, pdf=True, tex=False,figs = None):
    if len(name)==1:
        name = name[0]
    else:
        if len(name) % 2 == 1:
            raise ValueError('Number of arguments must be even')
        properties=OrderedDict()
        for j in range(len(name)//2):
            properties[name[2*j]]=name[2*j+1]
        name = _classify(properties)
    if tex:
        if figs is not None:
            raise ValueError('tex only supports saving current figure')
        matplotlib2tikz.save(name + '.tex')
    if pdf:
        if figs=='current':
            savefig(name + '.pdf', bbox_inches='tight')
        else:
            if figs is None or figs=='all':
                figs = plt.get_fignums()
            savenames = {n:name+'_{:d}.pdf'.format(n) for n in figs}
            for n in figs:
                plt.figure(n)
                savefig(savenames[n],bbox_inches='tight')
            from PyPDF2 import PdfFileMerger
            merger = PdfFileMerger()
            for n in figs:
                merger.append(open(savenames[n], 'rb'))
            with open(name+'.pdf', "wb") as fout:
                merger.write(fout)
            for n in figs:
                try:
                    os.remove(savenames[n])
                except:
                    pass
#@validate_args()
def plot_indices(mis, dims=None, weight_dict=None, N_q=1,labels=None):
    '''
    Plot multi-index set
    
    :param mis: Multi-index set
    :type mis: Iterable of SparseIndices
    :param dims: Which dimensions to use for plotting
    :type dims: List of integers.
    :param weight_dict: Weights associated with each multi-index
    :type weight_dict: Dictionary
    :param N_q: Number of percentile-groups plotted in different colors
    :type N_q: Integer>=1
    
    TODO: Change `weight_dict` to `weights`, exchange labels and dims, exchange N_q and dims
    '''
    if Dict.valid(mis):
        if labels is None or weight_dict is None:
            temp = list(mis.keys())
            if (List|Tuple).valid(temp[0]):
                if not (labels is None and weight_dict is None):
                    raise ValueError('mis cannot be dictionary with tuple entries if both labels and weight_dict are specified separately')
                weight_dict = {mi:mis[mi][0] for mi in mis}
                labels=  {mi:mis[mi][1] for mi in mis}
            else:
                if weight_dict is None:
                    weight_dict = mis
                else:
                    labels = mis
            mis = temp
        else:
            raise ValueError('mis cannot be dictionary if labels are specified separately')
    if dims is None:
        try:
            dims = len(mis[0])
        except TypeError:
            dims = sorted(list(set.union(*(set(mi.active_dims()) for mi in mis))))   
    if len(dims) > 3:
        raise ValueError('Cannot plot in more than three dimensions.')
    if len(dims) < 1:
        warnings.warn('Sure you don\'t want to plot anything?')
        return
    if weight_dict:
        values = list(weight_dict.values())
        weight_function = lambda mi: weight_dict[mi]
    else:
        if N_q > 1:
            raise ValueError('Cannot create percentile-groups without weight dictionary')
        weight_function = lambda mi: 1
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, N_q))  # @UndefinedVariable
    fig = plt.figure()#Creates new figure, because adding onto old axes doesn't work if they were created without 3d
    if len(dims) == 3:
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()
    ax.set_aspect('equal')
    size_function = lambda mi: sum([weight_function(mi2) for mi2 in mis if mi.equal_mod(mi2, lambda dim: dim not in dims)])
    sizes = {mi: np.power(size_function(mi), 0.1) for mi in mis}
    for q in range(N_q):
        if N_q > 1:
            plot_indices = [mi for mi in mis if weight_function(mi) >= np.percentile(values, 100 / N_q * q) and weight_function(mi) <= np.percentile(values, 100 / N_q * (q + 1))]
        else:
            plot_indices = mis
        X = np.array([mi[dims[0]] for mi in plot_indices])
        if len(dims) > 1:
            Y = np.array([mi[dims[1]] for mi in plot_indices])
        else:
            Y = np.array([0 for mi in plot_indices])
        if len(dims) > 2:
            Z = np.array([mi[dims[2]] for mi in plot_indices])
        else:
            Z = np.array([0 for mi in plot_indices])   
        sizes_plot = np.array([sizes[mi] for mi in plot_indices])
        if weight_dict:
            if len(dims) == 3:
                ax.scatter(X, Y, Z, s = 100 * sizes_plot / max(sizes.values()), color=colors[q], alpha=1)            
            else:
                ax.scatter(X, Y, s = 100 * sizes_plot / max(sizes.values()), color=colors[q], alpha=1)
        else:
            if len(dims) == 3:
                ax.scatter(X, Y, Z)
            else:
                ax.scatter(X, Y)
        try:
            max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
            if len(dims) == 3:
                for xb, yb, zb in zip(Xb, Yb, Zb):
                    ax.plot([xb], [yb], [zb], 'w')
            else:
                for xb, yb in zip(Xb, Yb):
                    ax.plot([xb], [yb], 'w')
        except ValueError:
            pass
        ax.set_xlabel('Dim. ' + str(dims[0]))
        if len(dims) > 1:
            ax.set_ylabel('Dim. ' + str(dims[1]))
        if len(dims) > 2:
            ax.set_zlabel('Dim. ' + str(dims[2]))
        plt.grid()
    if labels:
        for mi in labels:
            ax.annotate('{:.3g}'.format(labels[mi]),xy=(mi[0],mi[1]))
    if weight_dict:
        ax.legend([patches.Patch(color=color) for color in np.flipud(colors)],
                    ['{:.0f} -- {:.0f} percentile'.format(100*(N_q-1-i)/N_q,100*(N_q-i)/N_q) for i in reversed(range(N_q))])
    
def ezplot(f,xlim,ylim=None,ax = None,vectorized=True,N=None):
    '''
    Plot polynomial approximation.
    
    :param vectorized: `f` can handle an array of inputs
    '''
    d = 1 if ylim is None else 2
    show = False
    if ax is None:
        fig = plt.figure()
        show = True
        ax = fig.gca() if d==1 else fig.gca(projection='3d')
    if d == 1:
        if N is None:
            N = 200
        L = xlim[1] - xlim[0]
        X = np.linspace(xlim[0] + L / N, xlim[1] - L / N, N)
        X = X.reshape((-1, 1))
        if vectorized:
            Z = f(X)
        else:
            Z = np.array([f(x) for x in X])
        ax.plot(X, Z)
    elif d == 2:
        if N is None:
            N = 30
        T = np.zeros((N, 2))
        L = xlim[1] - xlim[0]
        T[:, 0] = np.linspace(xlim[0] + L / N, xlim[1] - L / N, N) 
        L = ylim[1] - ylim[0]
        T[:, 1] = np.linspace(ylim[0] + L / N, ylim[1] - L / N, N) 
        X, Y = meshgrid(T[:, 0], T[:, 1])
        Z = grid_evaluation(X, Y, f,vectorized=vectorized)
        ax.plot_surface(X, Y, Z)
    if show:
        plt.show()
    
def plot3D(X, Y, Z):
    '''
    Surface plot.
    
    Generate X and Y using, for example
          X,Y = np.mgrid[0:1:50j, 0:1:50j]
        or
          X,Y= np.meshgrid([0,1,2],[1,2,3]).
    
    :param X: 2D-Array of x-coordinates
    :param Y: 2D-Array of y-coordinates
    :param Z: 2D-Array of z-coordinates
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    light = LightSource(90, 90)
    illuminated_surface = light.shade(Z, cmap=cm.coolwarm)  # @UndefinedVariable
    Xmin = np.amin(X)
    Xmax = np.amax(X)
    Ymin = np.amin(Y)
    Ymax = np.amax(Y)
    Zmin = np.amin(Z)
    Zmax = np.amax(Z)
    ax.contourf(X, Y, Z, zdir='x', offset=Xmin - 0.1 * (Xmax - Xmin), cmap=cm.coolwarm, alpha=1)  # @UndefinedVariable
    ax.contourf(X, Y, Z, zdir='y', offset=Ymax + 0.1 * (Ymax - Ymin), cmap=cm.coolwarm, alpha=1)  # @UndefinedVariable
    ax.contourf(X, Y, Z, zdir='z', offset=Zmin - 0.1 * (Zmax - Zmin), cmap=cm.coolwarm, alpha=1)  # @UndefinedVariable
    ax.plot_surface(X, Y, Z, cstride=5, rstride=5, facecolors=illuminated_surface, alpha=0.5)
    plt.show()
    
def plot_divergence(times, values, name=None, title=None, divergence_type='algebraic', expect_values=None,
                    expect_times=None, plot_rate=None, p=2, preasymptotics=True, stagnation=False, marker=None, legend='lower right'):  
    plot_convergence(times, values, name, title, np.zeros(values[0].shape), divergence_type, expect_values, expect_times,
                     plot_rate, p, preasymptotics, stagnation, marker, legend)
    
def plot_convergence(times, values, name=None, title=None, reference='self', convergence_type='algebraic', expect_residuals=None,
                     expect_times=None, plot_rate='fit', p=2, preasymptotics=True, stagnation=False, marker=None,
                     legend='lower left'):
    '''
    Show loglog or semilogy convergence plot.
    
    Specify :code:`reference` if exact limit is known. Otherwise limit is 
    taken to be last entry of :code:`values`.
    
    Distance to limit is computed as RMSE (or analogous p-norm if p is specified)
    
    Specify either :code:`plot_rate`(pass number or 'fit') or 
    :code:`expect_residuals` and :code:`expect_times` to add a second plot with
    the expected convergence.
    
    :param times: Runtimes
    :type times: List of positive numbers
    :param values: Outputs
    :type values: List of arrays
    :param reference: Exact solution, or 'self' if not available
    :type reference: Array or 'self'
    :param convergence_type: Convergence type
    :type convergence_type: 'algebraic' or 'exponential'
    :param expect_residuals: Expected residuals
    :type expect_residuals: List of positive numbers
    :param expect_times: Expected runtimes
    :type expect_times: List of positive numbers
    :param plot_rate: Expected convergence order
    :type plot_rate: Real or 'fit'
    :param ignore: If reference is not provided, how many entries (counting
       from the end) should be ignored for the computation of residuals. 
    :type ignore: Integer.
    :param ignore_start: How many entries counting from start should be ignored.
    :type ignore_start: Integer.
    :return: fitted convergence order
    '''
    name = name or ''
    self_reference = (isinstance(reference,str) and reference=='self') #reference == 'self' complains when reference is a numpy array
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.gca().tick_params(labeltop=False, labelright=True, right=True, which='both')
    plt.gca().yaxis.grid(which="minor", linestyle='-', alpha=0.5)
    plt.gca().yaxis.grid(which="major", linestyle='-', alpha=0.6)
    c_ticks = 3
    ACCEPT_MISFIT = 0.1
    values, times = np.squeeze(values), np.squeeze(times)
    assert(times.ndim == 1)
    assert(len(times) == len(values))
    sorting = np.argsort(times)
    times = times[sorting]
    values = values[sorting]
    if plot_rate == True:
        plot_rate = 'fit'
    if self_reference:
        if len(times) <= 2:
            raise ValueError('Too few data points')
        limit = values[-1]
        limit_time = times[-1]
        times = times[0:-1]
        values = values[0:-1]
    else:
        limit = np.squeeze(reference)
        limit_time = np.Inf
    residuals = np.zeros(len(times))
    N = limit.size
    for L in range(len(times)):
        if p < np.Inf:
            residuals[L] = np.power(np.sum(np.power(np.abs(values[L] - limit), p) / N), 1. / p)  #
        else:
            residuals[L] = np.amax(np.abs(values[L] - limit))
    try:
        remove = np.isnan(times) | np.isinf(times) | np.isnan(residuals) | np.isinf(residuals) | (residuals == 0) | ((times == 0) & (convergence_type == 'algebraic'))
    except TypeError:
        print(times,residuals)
    times = times[~remove]
    if sum(~remove) < (2 if self_reference else 1):
        raise ValueError('Too few valid data points')
    residuals = residuals[~remove]
    if convergence_type == 'algebraic':
        x = np.log(times)
        limit_x = np.log(limit_time)
    else:
        x = times
        limit_x = limit_time
    #min_x = min(x)
    max_x = max(x)
    y = np.log(residuals)
    try:
        rate, offset, min_x_fit, max_x_fit = _fit_rate(x, y, stagnation, preasymptotics, limit_x, have_rate=False if (plot_rate == 'fit' or plot_rate is None) else plot_rate)
    except FitError as e:
        warnings.warn(str(e))
        plot_rate = False
        rate = None
    if self_reference:
        if rate >= 0:
            warnings.warn('No sign of convergence')
        else:
            real_rate = _real_rate(rate, l_bound=min_x_fit, r_bound=max_x_fit, reference_x=limit_x)
            if (real_rate is None or abs((real_rate - rate) / rate) >= ACCEPT_MISFIT):
                warnings.warn(('Self-convergence strongly affects plot and would yield misleading fit.')
                              + (' Estimated true rate: {}.'.format(real_rate) if real_rate else '')
                              + (' Fitted rate: {}.'.format(rate) if rate else ''))      
    if plot_rate:
        name += '(Fitted rate: {:.2g})'.format(rate)  #
        if convergence_type == 'algebraic':
            X = np.linspace(np.exp(min_x_fit), np.exp(max_x_fit), c_ticks)
            plt.loglog(X, np.exp(offset) * X ** rate, '--', color=color)
        else:
            X = np.linspace(min_x_fit, max_x_fit, c_ticks)
            plt.semilogy(X, np.exp(offset + rate * X), '--', color=color)
    max_x_data = max_x
    keep_1 = (x <= max_x_data)
    if convergence_type == 'algebraic':
        plt.loglog(np.array(times)[keep_1], np.array(residuals)[keep_1], label=name, marker=marker, color=color)
        plt.loglog(np.array(times), np.array(residuals), marker=marker, color=color, alpha=0.5)
    else:
        plt.semilogy(np.array(times)[keep_1], np.array(residuals)[keep_1], label=name, marker=marker, color=color)
        plt.semilogy(np.array(times), np.array(residuals), marker=marker, color=color, alpha=0.5)
    if expect_times is not None and expect_residuals is not None:
        plt.loglog(expect_times, expect_residuals, '--', marker=marker, color=color) 
    if name:
        plt.legend(loc=legend)
    if title:
        plt.title(title)
    return rate

def _keep(x, y, l_bound, r_bound):
    keep = (x >= l_bound) & (x <= r_bound)
    return x[keep], y[keep]

def _qof(x_original, y_original, l_bound, r_bound, have_rate, scale_x, scale_y, p=1, prefer_length=1):
    x, y = _keep(x_original, y_original, l_bound, r_bound)
    if len(x) >= 2:
        coeffs = _fit(x, y, l_bound, r_bound, have_rate)
        fitted_y = coeffs[1] + coeffs[0] * x
        w = _weights(x)
        return 5 * 2 ** (1 / p) * np.sum(w * np.abs((y - fitted_y) / scale_y) ** p) ** (1 / p) + prefer_length * (1 - (r_bound - l_bound) / scale_x)
    else:
        return np.Inf
    
def _truncate(x, y, where, l_bound, r_bound, have_rate, prefer_length, min_distance, max_move):
    min_x = min(x)
    max_x = max(x)
    bounds = np.linspace(min_x, max_x, 20)
    required_distance = min_distance * (max_x - min_x)
    bounds = bounds[(bounds > l_bound) & (bounds < r_bound)]
    if where == 'left':
        right_limit = min(r_bound - required_distance, min_x + max_move * (max_x - min_x))
        bounds = bounds[bounds < right_limit]
    if where == 'right':
        left_limit = max(l_bound + required_distance, max_x - max_move * (max_x - min_x))
        bounds = bounds[bounds > left_limit]
    scale_y = np.abs(max(y) - min(y))
    if bounds.size:
        scale_x = np.abs(max(x) - min(x))
        misfit = np.Inf * np.ones(bounds.shape)
        for i, bound in enumerate(bounds):
            if where == 'left':
                misfit[i] = _qof(x, y, bound, r_bound, have_rate, scale_x, scale_y, prefer_length=prefer_length)
            if where == 'right':
                misfit[i] = _qof(x, y, l_bound, bound, have_rate, scale_x, scale_y, prefer_length=prefer_length)
        i_min = np.argmin(misfit)
        if misfit[i_min] < np.Inf:
            return bounds[i_min]
    return l_bound if where == 'left' else r_bound

def _weights(x):
    w = 1 / 2 * ((x[1:-1] - x[0:-2]) + (x[2:] - x[1:-1]))
    if len(x) < 2:
        print(1)
    w = np.concatenate(((x[1] - x[0]).reshape(-1), w, (x[-1] - x[-2]).reshape(-1)))
    w = w / np.sum(w)
    return w

def _fit(x, y, l_bound, r_bound, have_rate):
    x, y = _keep(x, y, l_bound, r_bound)
    if len(x) < 2:
        raise FitError()
    if have_rate is False:
        coeffs = np.polyfit(x, y, deg=1, w=np.sqrt(_weights(x)))
    else:
        coeffs = have_rate, weighted_median(y - have_rate * x, _weights(x))
    return coeffs

class FitError(Exception):
    def __init__(self, cause=''):
        super().__init__(' '.join(['Could not fit rate.', cause]))
    
def _fit_rate(x, y, stagnation, preasymptotics, reference_x, have_rate):
    max_x = max(x)
    min_x = min(x)
    min_distance = 0.2 if len(x) > 10 else (0.5 if len(x) >= 5 else 0.8)
    attempts = 10
    l_bounds = {'init':min_x}
    r_bounds = {'init':max_x, 'selfconvergence':reference_x}
    r_bound = lambda : min(r_bounds.values())
    l_bound = lambda : max(l_bounds.values())
    for attempt in range(1, attempts + 1):
        if stagnation:
            r_bounds['stagnation'] = _truncate(x, y, 'right', l_bound=l_bound(), r_bound=max_x, have_rate=have_rate, prefer_length=0.5, min_distance=min_distance, max_move=1)
        if preasymptotics:
            l_bounds['preasymptotics'] = _truncate(x, y, 'left', r_bound=r_bound(), l_bound=min_x, have_rate=have_rate, prefer_length=1, min_distance=min_distance, max_move=attempt / attempts)
    observed_rate, offset = _fit(x, y, l_bound(), r_bound(), have_rate) 
    return observed_rate, offset, l_bound(), r_bound()
    
def _real_rate(observed_rate, l_bound, r_bound, reference_x):
    real_rate = None
    attempts = 10
    if observed_rate < -1e-12:
        def toy_residual(x, real_rate):  # -inf when x is too close to reference_x
            with np.errstate(divide='ignore'):
                return np.log(np.exp(real_rate * x) - np.exp(real_rate * reference_x))
        def observed_rate_fn(real_rate):
            observed_rate = (toy_residual(r_bound, real_rate) - toy_residual(l_bound, real_rate)) / (r_bound - l_bound)
            return observed_rate
        for _ in range(attempts):
            try:
                real_rate = scipy.optimize.bisect(lambda real_rate: observed_rate_fn(real_rate) - observed_rate, observed_rate, -1e-12, rtol=0.01)
                break
            except ValueError:  # Probably because r_bound is so close to reference_x that even the slowest rate would still give huge observed rates
                r_bound = max(l_bound + 0.1 * (r_bound - l_bound), r_bound / 2)
    return real_rate
        
def plot_nterm_convergence(coeffs):
    T = np.power(np.sort(coeffs, 0)[::-1], 2)
    times, values = range(1, len(T)), np.sqrt(np.sum(T) - np.cumsum(T)[:-1])
    return plot_convergence(times, values, reference=0, plot_rate='fit')
