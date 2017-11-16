import numpy
import unittest
from swutil.multiprocessing import EasyHPC, wrap_mpi
from swutil.decorators import print_runtime
#@MPIMC()
def rand(M):
    return numpy.random.rand(M,1)

def func(M):
    #ca=numpy.random.rand(M,1)
    ca=rand(M)
    for j in range(len(ca)):
        ca[j]=ca[j]**2
    return ca

@wrap_mpi
def external():
    pass
    
@EasyHPC(backend='MPI',input='count',output='many')
def f(M):
    external()
    return numpy.zeros(M)

@EasyHPC(backend='MPI',input='count',output='many')
def func_MPI(M):
    return func(M)

@EasyHPC(backend='MP',input='count',output='many')
def func_MP(M):
    return func(M)

def MPI(M):
    f=EasyHPC(backend='MPI',input='count',output='many')(func)
    return f(M)

def MP(M):
    f=EasyHPC(backend='MP',input='count',output='many')(func)
    return f(M)

@EasyHPC(backend='MPI',input='count',output='many')
def MPIMP(M):
    a=EasyHPC(backend='MPI',input='count',output='many')(func)
    return a(M)
    
@EasyHPC(backend='MPI',input='many',output='one',reduce=sum)
def MPIsum(elements):
    return sum(elements)

@EasyHPC(backend='MP',input='many',output='one',reduce=sum)
def MPsum(elements):
    return sum(elements)

class TestMultiprocessing(unittest.TestCase):
    def setUp(self):
        self.N=10000000  
    def testnoMP(self):
        func(self.N)  
    def testMPI(self):
        print(func_MPI(self.N).shape)
        print(MPI(self.N).shape)    
    def testMP(self):
        print(func_MP(self.N).shape)
        print(MP(self.N).shape)
    def testMPIMP(self):
        print(MPIMP(self.N).shape)
    def testWrapMPI(self):
        print(f(self.N).shape)
    def testMPIlists(self):
        print(MPIsum(numpy.array([2*i for i in range(self.N)])))
    def testMPlists(self):
        print(MPsum(numpy.array([2*i for i in range(self.N)])))   
              
if __name__ == "__main__":
    #print_runtime(unittest.main)(exit=False)
    suite=unittest.TestLoader().loadTestsFromName(name='test_multiprocessing.TestMultiprocessing.testMPIlists')
    unittest.TextTestRunner().run(suite)
