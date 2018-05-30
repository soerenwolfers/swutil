import numpy
import random
import string
import shutil
import re
import keyword
import readline
from datetime import timedelta

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
    Generate alphanumerical string. Hint: Check whether module tempfile has what you want, especially when you are concerned about race conditions
    '''
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def random_word(length,dictionary = False):#may return offensive words if dictionary = True
    '''
    Creates random lowercase words from dictionary or by alternating vowels and consonants
    
    The second method chooses from 85**length words.
    The dictionary method chooses from 3000--12000 words for 3<=length<=12
    (though this of course depends on the available dictionary)
    
    :param length: word length
    :param dictionary: Try reading from dictionary, else fall back to artificial words
    '''
    if dictionary:
        try:
            with open('/usr/share/dict/words') as fp:
                words = [word.lower()[:-1] for word in fp.readlines() if re.match('[A-Za-z0-9]{}$'.format('{'+str(length)+'}'),word)]
            return random.choice(words)
        except FileNotFoundError:
            pass
    vowels = list('aeiou')
    consonants = list('bcdfghklmnprstvwz')
    pairs = [(random.choice(consonants),random.choice(vowels)) for _ in range(length//2+1)] 
    return ''.join([l for p in pairs for l in p])[:length]

def string_from_seconds(seconds):
    '''
    Converts seconds into elapsed time string of form 
    
    (X days(s)?,)? HH:MM:SS.YY
    
    '''
    td = str(timedelta(seconds = seconds))
    parts = td.split('.')
    if len(parts) == 1:
        td = td+'.00'
    elif len(parts) == 2:
        td = '.'.join([parts[0],parts[1][:2]])
    return td
def input_with_prefill(prompt, text):
    '''
    https://stackoverflow.com/questions/8505163/is-it-possible-to-prefill-a-input-in-python-3s-command-line-interface
    '''
    def hook():
        readline.insert_text(text)
        readline.redisplay()
    try:
        readline.set_pre_input_hook(hook)
    except Exception:
        pass
    result = input(prompt)
    try:
        readline.set_pre_input_hook()
    except Exception:
        pass
    return result

def is_identifier(s):
    '''
    Check if string is valid variable name
    '''
    return s.isidentifier() and not keyword.iskeyword(s)

if __name__ == '__main__':
    for _ in range(100):
        print(random_word(length = 17,dictionary = True))
