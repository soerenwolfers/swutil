import numpy
import unittest
from swutil.hpc import EasyHPC, wrap_mpi
from swutil.decorators import print_runtime
#@MPIMC()
def rand(M):
    return numpy.random.rand(M,1)

def func(M):
    ca=rand(M)
    for j in range(len(ca)):
        ca[j]=ca[j]**2
    return ca

def split_interval(interval):
    (a,b)=interval
    return (a,(b+a)/2),((b+a)/2,b)

def my_sum(elements):
    return sum(elements)

def _integrate_x(interval,h=2**(-10)):
    (a,b)=interval
    N=max(1,(b-a)/h)
    x=numpy.linspace(a,b,N,endpoint=False)
    out = numpy.sum(h*x)
    return out
integrate_x=EasyHPC(backend='MPI',n_tasks='implicitly many',n_results='one',split_job=split_interval,reduce=my_sum)(_integrate_x)

@wrap_mpi
def external():
    pass
    
@EasyHPC(backend='MPI',n_tasks='count',n_results='many')
def f(M):
    external()
    return numpy.zeros(M)

@EasyHPC(backend='MPI',n_tasks='count',n_results='many')
def func_MPI(M):
    return func(M)

@EasyHPC(backend='MP',n_tasks='count',n_results='many')
def func_MP(M):
    return func(M)

def MPI(M):
    f=EasyHPC(backend='MPI',n_tasks='count',n_results='many')(func)
    return f(M)

def MP(M):
    f=EasyHPC(backend='MP',n_tasks='count',n_results='many')(func)
    return f(M)

@EasyHPC(backend='MPI',n_tasks='count',n_results='many')
def MPIMP(M):
    a=EasyHPC(backend='MPI',n_tasks='count',n_results='many')(func)
    return a(M)
    
@EasyHPC(backend='MPI',n_tasks='many',n_results='one',reduce=sum)
def MPIsum(elements):
    return sum(elements)

@EasyHPC(backend='MP',n_tasks='many',n_results='one',reduce=sum)
def MPsum(elements):
    return sum(elements)

class TestMultiprocessing(unittest.TestCase):
    def setUp(self):
        self.N=10000000  
    def test_implicitly_many(self):
        self.assertAlmostEqual(integrate_x([0,1]),1/2,delta=1e-2)
    def testnoMP(self):
        self.assertEqual(func(self.N).size,self.N)  
    def testMPI(self):
        self.assertAlmostEqual(func_MPI(self.N).size,self.N,delta=self.N/10)
        self.assertAlmostEqual(MPI(self.N).size,self.N,delta=self.N/10)    
    def testMP(self):
        self.assertAlmostEqual(func_MP(self.N).size,self.N,delta=self.N/10)
        self.assertAlmostEqual(MP(self.N).size,self.N,delta=self.N/10)
    def testMPIMP(self):
        self.assertAlmostEqual(MPIMP(self.N).size,self.N,delta=self.N/10)
    def testWrapMPI(self):
        self.assertAlmostEqual(f(self.N).size,self.N,delta=self.N/10)
    def testMPIlists(self):
        self.assertEqual(MPIsum(numpy.array([2*i for i in range(self.N)])),self.N*self.N-self.N)
    def testMPlists(self):
        self.assertEqual(MPsum(numpy.array([2*i for i in range(self.N)])),self.N*self.N-self.N)   
              
if __name__ == "__main__":
    print_runtime(unittest.main)(exit=False)
    #suite=unittest.TestLoader().loadTestsFromName(name='test_multiprocessing.TestMultiprocessing.test_implicitly_many')
    #unittest.TextTestRunner().run(suite)
