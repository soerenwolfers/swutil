import builtins
import six
import numbers
import ast
import re
def merge_dicts(d1, d2):
    if any(d1[k] != d2[k] for k in d1.keys() & d2):
        raise ValueError('Cannot merge dictionaries')
    return dict(d1, **d2)
class Validator(object):
    def forgive(self,arg,level):
        return arg
    def validate(self,arg,forgiveness):
        if self.valid(arg):
            return arg
        for level in range(forgiveness):
            arg=self.forgive(arg=arg,level=level)
            if self.valid(arg):
                return arg   
        raise ParseError()
    
class ParseError(Exception):
    pass
class Args(object):
    @staticmethod
    def removeNestedParentheses(s):
        ret = ''
        skip = 0
        for i in s:
            if i == '(':
                skip += 1
            elif i == ')'and skip > 0:
                skip -= 1
            elif skip == 0:
                ret += i
        return ret
    def __init__(self,args_dict,specs=None,**kwspecs):
        specs=specs or dict()
        specs=merge_dicts(specs,kwspecs)
        self.__dict__.update(args_dict)
        def get_callable(spec):
            if isinstance(spec,six.string_types):
                spec=self.removeNestedParentheses(spec)
                if not ' ' in spec and len(spec)>0:
                    if 'v_' in spec:#Only verify spec
                        cl=spec.split('v_',1)[1].lower()
                        forgiveness=0
                    elif 'f_' in spec:#Force conversion
                        cl=spec.split('f_',1)[1].lower()
                        forgiveness=2
                    else:#Reasonable conversions
                        cl=spec
                        forgiveness=1
                    if hasattr(self,cl):
                        ob=getattr(self,cl)()
                        return lambda arg: ob.validate(arg,forgiveness)
                    else:
                        raise ValueError('Could not understand specification {}'.format(spec))
            elif hasattr(spec,'__call__'):
                return spec
            return False
        for arg in specs:
            if isinstance(specs[arg],(builtins.tuple,builtins.list)):
                specs[arg],default=specs[arg]
                if not arg in self.__dict__:
                    self.__dict__[arg]=default
            else:
                if not arg in self.__dict__:
                    self.__dict__[arg]=None
            if get_callable(specs[arg]):#is either an understood string or a function
                arg_specs=[specs[arg]]
            else:#it is a string of multiple words
                arg_specs=self.removeNestedParentheses(specs[arg]).split()[::-1]
            for spec in arg_specs:
                try:
                    func=get_callable(spec)
                    self.__dict__[arg]=func(self.__dict__[arg])
                except ParseError:
                    msg='{} was rejected by {}'.format(
                        self.__dict__[arg].__repr__(),specs[arg])
                    if len(arg_specs)>1:
                        msg+=' ({} was rejected by {})'.format(
                            self.__dict__[arg],spec)
                    raise ValueError(msg) from None
        
    class bool(Validator):
        def valid(self,arg):
            return isinstance(arg,builtins.bool)
        def forgive(self,arg,level):
            if level==0:
                if arg in ['True','true']:
                    return True
                if self.arg in ['False','false']:
                    return False
            if level==1:
                return builtins.bool(arg)

    class string(Validator):
        def valid(self,arg):
            return isinstance(arg,six.string_types)
        def forgive(self,arg,level):
            if level==0:
                if isinstance(arg,numbers.Number) and type(arg)!=builtins.bool:
                    return str(arg)
            if level==1:
                return builtins.str(arg)

    class nonnegative(Validator):
        def valid(self,arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            try:
                return arg>=0
            except Exception:
                return False
        def forgive(self,arg,level):
            if level==0:
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    return arg
            elif level==1:
                try:
                    return max(arg,0)
                except Exception:
                    return arg
    class positive(Validator):
        def valid(self,arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            try:
                return arg>0
            except Exception:
                return False
        def forgive(self,arg,level):
            if level==0:
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    return arg
    class nonpositive(Validator):
        def valid(self,arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            try:
                return arg<=0
            except Exception:
                return False
        def forgive(self,arg,level):
            if level==0:
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    pass
            elif level==1:
                try:
                    return min(arg,0)
                except Exception:
                    pass
    class negative(Validator):
        def valid(self,arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            try:
                return arg<0
            except Exception:
                return False
        def forgive(self,arg,level):
            if level==0:
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    pass 
    class float(Validator):
        def valid(self,arg):
            return isinstance(arg,builtins.float)
        def forgive(self,arg,level):
            if level==0:
                return float(arg)
    class integer(Validator):
        def valid(self,arg):
            return isinstance(arg,builtins.int)
        def forgive(self,arg,level):
            if level==0:
                if isinstance(arg,six.string_types):
                    return ast.literal_eval(arg)
                if isinstance(arg,builtins.float) and arg.is_integer():
                    return int(arg)
                return arg
            if level==1:
                return int(arg)
    class function(Validator):
        def valid(self,arg):
            return hasattr(arg,'__call__')
        def forgive(self,arg,level):
            if level==0:
                return lambda x:arg
    class dict(Validator):
        def valid(self,arg):
            return isinstance(arg,builtins.dict)
        def forgive(self,arg,level):
            if level==0:
                if isinstance(arg,(builtins.list,builtins.tuple)):
                    d=builtins.dict()
                    for i in range(len(arg)):
                        d[i]=arg[i]
                    return d
    class set(Validator):
        def valid(self,arg):
            return isinstance(arg,builtins.set)
        def forgive(self,arg,level):
            if level==0:
                if isinstance(arg,(builtins.list,builtins.tuple)):
                    return set(arg)
            if level==1:
                return {arg}
def test(a,b,c=1,d=-2.4):
    args=Args(locals(),d='float',b='f_dict',c='set')
    print(args.a,args.b,args.d,args.c)
if __name__=='__main__':
    test(1e3,[2],[3])