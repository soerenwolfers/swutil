import numpy
import random
import string
class NotPassedClass():
    def __bool__(self):
        return False
    def __str__(self):
        return '<NotPassed>'
    def __repr__(self):
        return '<NotPassed>'
    def __eq__(self,other):
        return isinstance(other,NotPassedClass)
    def __req__(self,other):
        return isinstance(other,NotPassedClass)
    def __call__(self,other):
        return isinstance(other,NotPassedClass)
    
NotPassed=NotPassedClass()
def Passed(other):
    return not NotPassed(other)

def chain(*fs):
    def chained(x):
        for f in reversed(fs):
            if f:
                x=f(x)
        return x
    return chained

def split_list(l,N):
    npmode = isinstance(l,numpy.ndarray)
    if npmode:
        l=list(l)
    L=int(len(l)/N)
    s=[]
    i=0
    for n in range(N):
        s.append(l[i:i+L])
        i=i+L
    for n in range(N):
        s[n]+=l[i:i+1]
        i=i+1
    if npmode:
        s=numpy.array(s)
    return s

def random_string(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))
    
if __name__=='__main__':
    for i in range(1,14):
        print(i,split_list(list(range(12)),i))