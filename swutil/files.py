import os
import fnmatch

def find_files(pattern, path=None):
    '''
    https://stackoverflow.com/questions/1724693/find-a-file-in-python
    '''
    if not path:
        path = os.getcwd()
    result = []
    for root, __, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(os.path.join(root,name),pattern):
                result.append(os.path.join(root, name))
    return result

def find_directories(pattern, path=None):
    if not path:
        path = os.getcwd()
    result = []
    for root, __, __ in os.walk(path):
        if fnmatch.fnmatch(root,pattern):
            result.append(root)
    return result

def delete_empty_files(fpaths):  
    for fpath in fpaths:
        try:
            if os.path.isfile(fpath) and os.path.getsize(fpath) == 0:
                os.remove(fpath)
        except Exception:
            pass
        
def delete_empty_directories(dirs):
    for d in dirs:
        try:
            os.rmdir(d)
        except OSError: # raised if not empt
            pass
    

def append_text(file_name, text):
    if text:
        with open(file_name, 'a') as fp:
            fp.write(text)