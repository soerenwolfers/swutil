import numpy
import random
import string
import shutil

class no_context():
    def __enter__(self, *args):
        pass
    def __exit__(self, *args):
        pass
import inspect

def isdebugging():
    for frame in inspect.stack():
        if frame[1].endswith('pydevd.py') or frame[1].endswith('pdb.py'):
            return True
    return False
  
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

def string_dialog(title,label):
    import tkinter
    import tkinter.simpledialog
    root = tkinter.Tk()
    root.withdraw()
    return tkinter.simpledialog.askstring(title, label)

def cmd_exists(cmd):
    '''
    Check whether given command is available on system
    '''
    return shutil.which(cmd) is not None
def split_integer(N,bucket = None, length = None):
    if bucket and not length:
        if bucket <1:
            raise ValueError()
        length = N//bucket + (1 if N%bucket else 0)
    if length ==0:
        if N ==0:
            return []
        else:
            raise ValueError()
    tmp = numpy.array([N//length]*length)
    M = N % length
    tmp[:M]+=1
    return list(tmp)

def split_list(l,N):
    '''
    Subdivide list into N lists
    '''
    npmode = isinstance(l,numpy.ndarray)
    if npmode:
        l=list(l)
    g=numpy.concatenate((numpy.array([0]),numpy.cumsum(split_integer(len(l),length=N))))
    s=[l[g[i]:g[i+1]] for i in range(N)]
    if npmode:
        s=[numpy.array(sl) for sl in s]
    return s

def random_string(length):
    '''
    Generate alphanumerical string. Hint: Check if module tempfile has what you want, especially when you are concerned about race conditions
    '''
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
        