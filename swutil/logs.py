from _io import BytesIO
import sys
import datetime

class Capturing(list):
    '''
    From http://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    '''
    def __enter__(self):
        self._stdout,self._stderr = sys.stdout,sys.stderr
        sys.stdout, sys.stderr = self._stdio,self._errio = BytesIO(),BytesIO()
        return self
    def __exit__(self, *args):
        self.extend([self._stdio.getvalue(),self._errio.getvalue()])
        del self._stdio,self._errio    # free up some memory
        sys.stdout,sys.stderr = self._stdout,self._stderr
        
        
class Log(object):   
    def __init__(self,print_filter=True,write_filter=False,file_name=None,lock=None):
        if print_filter is True:
            print_filter = lambda _: True
        if print_filter is False:
            print_filter = lambda _: False
        self.print_filter=print_filter
        self.file_name=file_name
        if write_filter and not self.file_name:
            raise ValueError('Specify file_name to write log in file')
        if write_filter is True:
            write_filter = lambda _: True
        if write_filter is False:
            write_filter = lambda _: False
        self.write_filter=write_filter
        self.entries=[]
        self.lock=lock
        
    def log(self,message=None,group=None,tags=None):
        if self.lock:
            self.lock.acquire()
        entry=Entry(group=group,message=message,tags=tags)
        self.entries.append(entry)
        if self.print_filter(entry):
            print(entry)
        if self.write_filter(entry):
            with open(self.file_name,'a') as fp:
                fp.write(str(entry)+'\n')
        if self.lock:
            self.lock.release()
    
    def print_log(self,print_filter=None):
        print(self.__str__(print_filter))
      
    def __str__(self,print_filter=None):
        print_filter=print_filter or self.print_filter
        return '\n'.join([str(entry) for entry in self.entries if print_filter(entry)])

        
#@staticmethod            
def filter_generator(require_group=None,require_tags=None,require_message=None,exclude_group=None,exclude_tags=None,exclude_message=None):
    def filter(entry):  # @ReservedAssignment
        if require_group is None:
            require_group=[]
        else:
            if not type(require_group) is list:
                require_group=[require_group]
        if require_tags is None:
            require_tags=[]
        else:
            if not type(require_tags) is list:
                require_tags=[require_tags]
        if require_message is None:
            require_message=[]
        else:
            if not type(require_message) is list:
                require_message=[require_message]         
        if exclude_group is None:
            exclude_group=[]
        else:
            if not type(exclude_group) is list:
                exclude_group=[exclude_group]
        if exclude_tags is None:
            exclude_tags=[]
        else:
            if not type(exclude_tags) is list:
                exclude_tags=[exclude_tags]
        if exclude_message is None:
            exclude_message=[]
        else:
            if not type(exclude_message) is list:
                exclude_message=[exclude_message]
        return  (
                    (any([group == entry.group for group in require_group]) or not require_group)
                    and 
                    all([tag in entry.tags for tag in require_tags])
                    and 
                    all([message in entry.message for message in require_message])
                    and
                    not entry.group in exclude_group
                    and
                    all([tag not in entry.tags for tag in exclude_tags])
                    and
                    all([message not in entry.message for message in exclude_message])
                )
    return filter
    
class Entry(object):
    def __init__(self,group=None,message=None,tags=None):
        if group is None:
            self.group=''
        else:
            self.group=group
        if message is None:
            self.message=''
        else:
            self.message=message
        if tags is None:
            self.tags=[]
        else:
            self.tags=tags
        self.time=datetime.datetime.now()
        
    def __str__(self):
        string='<'+str(self.time)
        if self.group:
            string+= ' | '+ self.group
        string+='> '+self.message
        return string