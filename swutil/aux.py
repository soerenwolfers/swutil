import numpy
import random
import string
import shutil

def chain(*fs):
    '''
    Concatenate functions
    '''
    def chained(x):
        for f in reversed(fs):
            if f:
                x=f(x)
        return x
    return chained

def cmd_exists(cmd):
    '''
    Check whether given command is available on system
    '''
    return shutil.which(cmd) is not None

def split_list(l,N):
    '''
    Subdivide list into N lists
    '''
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
    '''
    Generate alphanumerical string. Hint: Check if module tempfile has what you want, especially when you are concerned about race conditions
    '''
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
        