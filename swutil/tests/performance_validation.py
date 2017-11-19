from swutil.validation import Positive,  Negative, validate_args
from swutil.decorators import print_runtime
from random import random
import numpy
K=100
def f(a:Positive,b:Negative):  
    M=numpy.ones((K,K))
    x=numpy.random.rand(K)
    M.dot(x)
    return a+b
validated_f = validate_args()(f)
N=100000
@print_runtime
def run_non_validated():
    for _ in range(N):
        a,b=random(),-random()
        f(a,b)
@print_runtime
def run_validated():
    for _ in range(N):
        a,b=random(),-random()
        validated_f(a,b)
run_non_validated()
#run_validated()
#validated_f.validate=False
run_validated()