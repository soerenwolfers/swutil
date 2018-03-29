import dill
import numpy
from swutil.aux import split_list, random_string, chain, cmd_exists
import math
import itertools
import os
import inspect
import time
import sys
import subprocess
import pathos
import argparse

from swutil.validation import In, Bool, Function, validate_args, Passed, NotPassed
#TODO: Add optional logging
def wrap_mpi(f):
    info = argparse.Namespace()
    info.wrap_MPI = True
    return MultiProcessorWrapper(f, _MPI_processor, _MPI_finalizer, info)   
@validate_args(warnings=False)
class EasyHPC(object):#DecoratorFactory
    def __init__(self,
                 backend:In('MP', 'MPI')='MP',
                 n_tasks:In('implicitly many', 'many', 'one', 'count')='one',#Count is special case of implicitly many where it is already known how to split jobs 
                 n_results:In('many', 'one')='one',
                 aux_output:Bool=True,  # Parellelize only first entry of n_results is tuple
                 reduce:Function=NotPassed,
                 split_job=NotPassed
                 ):
        if backend == 'MPI': 
            self.processor = _MPI_processor
            self.finalizer = _MPI_finalizer
        if backend == 'MP':
            self.processor = _MP_processor
            self.finalizer = None
        self.info = argparse.Namespace()
        self.info.n_tasks = n_tasks
        self.info.n_results = n_results
        self.info.reduce = reduce
        self.info.wrap_MPI = False
        self.info.aux_output = aux_output 
        self.info.split_job = split_job
        if self.info.n_tasks == 'implicitly many':
            if self.info.n_results == 'many':
                raise ValueError('Do not know how to handle functions that handle implicitly many tasks and return multiple results')
            if NotPassed(self.info.split_job):
                raise ValueError('Functions handling implicitly many tasks must specfiy how to split a job using `split_job`')
        if self.info.n_results == 'one':
            if NotPassed(self.info.reduce):
                raise ValueError('Functions that return single results must specify how to reduce multiple results using `reduce`')
    def __call__(self, f):
        return MultiProcessorWrapper(f, self.processor, self.finalizer, self.info)

class MultiProcessorWrapper(object):
    def __init__(self, f, processor, finalizer, info):
        self.f = f
        self.processor = processor
        self.finalizer = finalizer
        self.info = info
    def __call__(self, *args, **kwargs):  
        if '__second_call' in kwargs:
            kwargs.pop('__second_call')
            return self.f(*args, **kwargs)
        else:
            if args:
                M, args = args[0], args[1:]
            else:
                if not self.info.wrap_MPI:
                    raise ValueError('Must specify task(s)')
                M, args = NotPassed, NotPassed               
            f_path = inspect.getsourcefile(self.f)
            f_name = self.f.__name__
            ID = '.easyhpc_' + f_name + '_' + str(time.time()) + '_' + random_string(8)
            r, ID = self.processor(args, kwargs, M, self.f, f_path, f_name, ID, self.info)
            out = _reduce_first_output(self.info, r)
            if self.finalizer:
                self.finalizer(ID)
            return out
def _MPI_processor(args, kwargs, M, f, f_path, f_name, ID, info):
    from mpi4py import MPI
    if info.wrap_MPI:
        child = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=['-c', 'import swutil.multiprocessing;swutil.multiprocessing._MPI_worker(mpi_child=True)'],
            maxprocs=1)
        child.bcast((args, kwargs, M, f_path, f_name, info), root=MPI.ROOT)
        r = child.recv(source=MPI.ANY_SOURCE)
        child.Disconnect()
    elif MPI.COMM_WORLD.Get_size() == 1 and MPI.Comm.Get_parent() == MPI.COMM_NULL:
        if cmd_exists('srun'):
            mpi_executable = 'srun'
        else:
            mpi_executable = 'mpiexec'
        command = ' '.join((mpi_executable,sys.executable,'-c "import swutil.multiprocessing;swutil.multiprocessing._MPI_worker(ID=\'{}\')"'))
        command = command.format(ID)
        with open(ID + '_task', 'wb') as file:
            dill.dump((args, kwargs, M, f_path, f_name, info), file)
        subprocess.check_call(command, shell=True) 
        with open(ID + '_results', 'rb') as file:
            r = dill.load(file)
    else:
        r = _MPI_worker(pure_python=(args, kwargs, M, f_path, f_name, info))
    return r, ID

def _MP_processor(args, kwargs, M, f, f_path, f_name, ID, info):
    N = pathos.multiprocessing.cpu_count()
    p = pathos.multiprocessing.Pool(N)
    r = p.map(_MP_worker, [(f_path, f_name, args, kwargs, M, n, N, info) for n in range(N)])
    return r, ID

def _MPI_worker(ID=None, pure_python=False, mpi_child=False):
    from mpi4py import MPI
    if mpi_child:
        parent = MPI.Comm.Get_parent()
        (args, kwargs, M, f_filename, f_name, info) = parent.bcast(None, root=0)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        N = comm.Get_size()
    elif pure_python:
        (args, kwargs, M, f_filename, f_name, info) = pure_python
        N = 1
        rank = 0
    else:
        with open(ID + '_task', 'rb') as file:
            (args, kwargs, M, f_filename, f_name, info) = dill.load(file)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        N = comm.Get_size()
    rr = _common_work(f_filename, f_name, M, N, rank, args, kwargs, info)
    if mpi_child:
        r = comm.gather(rr, 0)
        if comm.Get_rank() == 0:
            parent.send(r, dest=0)
        parent.Disconnect()  
    elif pure_python:
        r = [rr]
        return r
    else:
        r = comm.gather(rr, 0)
        if comm.Get_rank() == 0:
            with open(ID + '_results', 'wb') as file:
                dill.dump(r, file)
        
def _MP_worker(arg):
    (f_filename, f_name, args, kwargs, M, rank, N, info) = arg
    numpy.random.seed(rank)#TODO replace by randomint+rank
    return _common_work(f_filename, f_name, M, N, rank, args, kwargs, info) 
           
def _common_work(f_filename, f_name, M, N, rank, args, kwargs, info):
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", f_filename)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    f = getattr(foo, f_name)
    if info.wrap_MPI:
        jobs = [M] * N
    else:
        if info.n_tasks == 'count':
            jobs = [int(math.ceil(M / N))] * N
        elif info.n_tasks in ('many', 'one'):
            jobs = split_list(M, N)
        elif info.n_tasks == 'implicitly many':
            jobs = [M]
            while len(jobs) < N:
                if 2*len(jobs)<=N:
                    jobs = list(itertools.chain(*[info.split_job(job) for job in jobs]))
                else:
                    need=N-len(jobs)
                    jobs=list(itertools.chain(*([info.split_job(job) for job in jobs[:need]]+[jobs[need:]])))
    if isinstance(f, MultiProcessorWrapper):
        kwargs['__second_call'] = True
    if not info.wrap_MPI and info.n_tasks == 'one':
        if Passed(M):
            return  [f(job, *args, **kwargs) for job in jobs[rank]]
        else:
            return [f(**kwargs) for job in jobs[rank]]
    else:
        if Passed(M):
            out= f(jobs[rank], *args, **kwargs)
            #print(jobs[rank],out)
            return out
        else:
            return f(**kwargs)


def _reduce_first_output(info, r):
    aux_output = False if info.wrap_MPI else (info.aux_output and isinstance(r[0], tuple))
    numpy_mode = isinstance(r[0], numpy.ndarray) or aux_output and isinstance(r[0][0], numpy.ndarray)
    concatenate = numpy.concatenate if numpy_mode else lambda r: list(itertools.chain(*r))
    if info.wrap_MPI:
        reduce = lambda r: r[0]
    else:
        if info.n_tasks == 'one' and info.n_results == 'one':  # Each entry of r is list
            reduce = chain(info.reduce, concatenate)
        if info.n_tasks in ('implicitly many', 'many', 'count') and info.n_results == 'one':  # Each entry of r is singleton. Need reduce
            reduce = info.reduce  # maybe f should be used as info.reduce by default if info.n_tasks is 'many'. 
                                # probably jobs should be passed to info.reduce if info.n_tasks is 'implicitly many'
        if info.n_tasks in ('implicitly many', 'many', 'count') and info.n_results == 'many':  # In future, first apply reduce to each element of r and then to resulting r
            reduce = chain(info.reduce, concatenate)
    if aux_output:
        return (reduce([rr[0] for rr in r]),) + r[0][1:]
    else:
        return reduce(r)

def _MPI_finalizer(ID): 
    try:
        os.remove(ID + '_results')
    except FileNotFoundError:
        pass
    try:
        os.remove(ID + '_task') 
    except FileNotFoundError:
        pass
