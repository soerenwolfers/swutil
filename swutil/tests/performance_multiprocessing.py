from swutil.decorators import print_runtime
from swutil.multiprocessing import EasyHPC
from swutil.tests.test_multiprocessing import split_interval, _integrate_x, func,\
    func_MP, func_MPI



if __name__=='__main__':
    MPI_f = (print_runtime(EasyHPC(backend='MPI',n_tasks='implicitly many',n_results='one',reduce=sum,split_job=split_interval)(_integrate_x)))
    MP_f = (print_runtime(EasyHPC(backend='MP',n_tasks='implicitly many',n_results='one',reduce=sum,split_job=split_interval)(_integrate_x)))
    interval=[0,1]
    h=2**(-28)
    print(print_runtime(_integrate_x)(1,h))
    print(MPI_f(interval,h))
    print(MP_f(interval,h))
    print('..................')
    N=int(1e6)
    print_runtime(func)(N)
    print_runtime(func_MP)(N)
    print_runtime(func_MPI)(N)
