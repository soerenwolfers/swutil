import six
import ast
from swutil.aux import NotPassed
import builtins
import numbers
from math import inf
from swutil.decorators import find_implementation
import numpy
from swutil.decorators import memoize

#Check function specifications upon call
class Specification(object):
    def __init__(self,forgiveness=1):
        self.forgiveness=forgiveness
    def valid(self,arg):
        if self.forgiveness:
            return True
        else:
            return False
    def forgive(self,arg,level):
        return arg
    
class OrSpecification(Specification):
    def __init__(self,specifications):
        self.specifications=specifications
        self.forgiveness=max(specification.forgiveness if hasattr(specification,'forgiveness') else 0 for specification in self.specifications)

    def valid(self,arg):
        for specification in self.specifications:
            try:
                if specification.valid(arg):
                    return True
            except Exception:
                pass
            
class SpecificationTree(object):
    def __init__(self,value_specification=None,key_specification=None,value_specifications=[]):
        self.value_specification=value_specification
        self.value_specifications=value_specifications
        self.key_specification=key_specification
        self.leaves=[]
        self.or_mode=False
    def add_leaf(self,leaf):
        if self.or_mode:
            if not isinstance(self.leaves[-1],OrSpecificationTree):
                self.leaves[-1]=OrSpecificationTree(self.leaves[-1])
            self.leaves[-1].ors.append(leaf)
            self.or_mode=False
        else:
            self.leaves.append(leaf)
        
        
class OrSpecificationTree(object):
    def __init__(self,tree):
        self.ors=[tree]
        self.value_specification=None
        self.key_specification=None
        
class SpecificationLeaf(object):
    def __init__(self,name,specification):
        self.specification=specification
        self.value_specification=None
        self.key_specification=None
        self.name=name
        
class MalformedStringError(Exception):
    pass

def _get_tuple(s):
    if not s[0]=='(':
        return '',0
    in_string=False
    open=0
    for j,c in enumerate(s):
        if c=='(' and not in_string:
            open+=1
        if c=='"':
            if in_string=='"':
                in_string=False
            elif not in_string:
                in_string='"'
        if c=="'":
            if in_string=="'":
                in_string=False
            elif not in_string:
                in_string="'"
        if c==')' and not in_string:
            open-=1
        if open==0:
            if s[j-1]!=',':
                ret=s[:j]+','+')'
            else:
                ret=s[:j+1]
            return ret,j+1
    raise MalformedStringError
@memoize
def get_specification_tree(spec,aux=None):
    aux=aux or []
    aux=builtins.list(aux)
    stack=[]
    if isinstance(spec,six.string_types):
        spec=spec.strip()
        spec_in=spec
        controls=['{','}',':','|',',']
        cl=None
        stack.append(SpecificationTree())
        while spec:
            if spec[0] in controls:
                if spec[0]=='{':
                    stack.append(SpecificationTree())
                if spec[0]=='}':
                    tmp=stack.pop()
                    if tmp.value_specification:
                        new_key_spec=tmp.value_specification
                        new_value_spec=SpecificationTree()
                        new_value_spec.leaves=tmp.leaves
                        tree=SpecificationTree(value_specification=new_value_spec,key_specification=new_key_spec)
                        ob=standards.allows_spec(0,'items')
                        leaf=SpecificationLeaf(name='allows("items")',specification=ob)
                        tree.add_leaf(leaf)
                    elif tmp.value_specifications:
                        new_value_spec=SpecificationTree()
                        new_value_spec.leaves=tmp.leaves
                        tree=SpecificationTree(value_specifications=tmp.value_specifications+[new_value_spec])
                    else:
                        tree=SpecificationTree(value_specification=tmp)
                        ob=standards.iterable_spec(0)
                        leaf=SpecificationLeaf(name='iterable',specification=ob)
                        tree.add_leaf(leaf)
                    stack[-1].add_leaf(tree)
                if spec[0]==':':
                    tmp=stack.pop()
                    if tmp.value_specifications:
                        raise ValueError('Cannot use different specifications for different keys of dictionary')
                    stack.append(SpecificationTree(value_specification=tmp))
                if spec[0]==',':
                    tmp=stack.pop()
                    if tmp.value_specification:
                        raise ValueError('Cannot use different specifications for different values of dictionary')
                    new_value_spec=SpecificationTree()
                    new_value_spec.leaves=tmp.leaves
                    tree=SpecificationTree(value_specifications=tmp.value_specifications+[new_value_spec])
                    stack.append(tree)
                if spec[0]=='|':
                    stack[-1].or_mode=True
                spec=spec[1:].strip()
            else:
                tmp_specs=[]
                total_name=''
                end=False
                while not end:
                    terminators=['*','(',' ','}',':','|',',']
                    length=len(spec)
                    for i,ch in enumerate(spec):
                        if ch in terminators:
                            length=i
                            terminator=ch
                            break
                    name=spec[:length]
                    if length!=len(spec) and terminator in ['*','(']:
                        if terminator=='*':
                            skip=length+1
                            if aux:
                                cl_args=aux.pop(0)
                            else:
                                raise ValueError('Specification {}* needs additional arguments'.format(name))
                        elif terminator=='(':
                            tuple_string,skip=_get_tuple(spec[length:])
                            list_string='['+tuple_string[1:-1]+']'
                            cl_args=ast.literal_eval(list_string)
                            skip=length+skip
                        if not isinstance(cl_args,(builtins.list,builtins.tuple)):
                            cl_args=[cl_args]
                        else:
                            cl_args=builtins.list(cl_args)
                    else:
                        skip=length
                        cl_args=builtins.list()
                    if name.startswith('v-'):#Only verify spec
                        cl=name.split('v-',1)[1].lower()
                        forgiveness=0
                    elif name.startswith('f-'):#Force conversion
                        cl=name.split('f-',1)[1].lower()
                        forgiveness=2
                    else:#Reasonable conversions
                        cl=name.lower()
                        forgiveness=1
                    if cl=='any':
                        ob=Specification(1)
                    elif cl=='none':
                        ob=Specification(0) 
                    elif cl=='complies':
                        if len(cl_args)==1:
                            ob=cl_args[0]
                        else:
                            raise ValueError('Need to pass single SpecificationTree to comply with')
                    elif hasattr(standards,cl+'_spec'):
                        try:
                            ob=getattr(standards,cl+'_spec')(*([forgiveness]+cl_args))
                        except TypeError:
                            raise ValueError('Specification \'{}\' could not be instantiated with settings \'{}\''.format(cl,cl_args)) from None
                    else:
                        raise ValueError('Did not understand specification {}'.format(name))
                    if len(spec)>skip and spec[skip]=='|':
                        skip+=1
                    else:
                        end=True
                    total_name+=spec[:skip]
                    tmp_specs.append(ob)
                    spec=spec[skip:].strip()
                if len(tmp_specs)>1:
                    ob=OrSpecification(tmp_specs)
                else:
                    ob=tmp_specs[0]    
                spec_leaf=SpecificationLeaf(name=total_name,specification=ob)
                stack[-1].add_leaf(spec_leaf)
        if len(stack)>1:
            stack[-2].add_leaf(stack.pop())
        if len(stack)!=1:
            raise ValueError('Did not understand specification {}'.format(spec_in))
        if len(aux)>0:
            raise ValueError('{} was passed as part of specification but never used'.format(aux))
        spec_tree=stack[0]
    elif isinstance(spec,type):
        spec_tree=get_specification_tree('class*', [spec])
    elif hasattr(spec,'__call__'):
            spec_tree=SpecificationLeaf(name=spec.__name__,specification=spec)
    elif isinstance(spec,SpecificationTree):
        spec_tree=spec
    elif isinstance(spec,Specification):
        spec_tree=SpecificationLeaf(name=spec.__name__,specification=spec)
    else:
        raise ValueError('Did not understand specification {}'.format(spec))
    return spec_tree
    
class standards(object):
    class bool_spec(Specification):
        @staticmethod
        def valid(arg):
            return isinstance(arg,builtins.bool)
        @staticmethod
        def forgive(arg,level):
            if level==0:
                if arg in ['True','true']:
                    return True
                if arg in ['False','false']:
                    return False
            if level==1:
                return builtins.bool(arg)

    class string_spec(Specification):
        @staticmethod
        def valid(arg):
            return isinstance(arg,six.string_types)
        @staticmethod
        def forgive(arg,level):
            if level==0:
                if isinstance(arg,numbers.Number) and type(arg)!=builtins.bool:
                    return str(arg)
            if level==1:
                return builtins.str(arg)

    class nonnegative_spec(Specification):
        @staticmethod
        def valid(arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            try:
                return arg>=0
            except Exception:
                return False
        @staticmethod
        def forgive(arg,level):
            if level==0:
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    pass
            elif level==1:
                try:
                    return max(arg,0)
                except Exception:
                    pass
    class positive_spec(Specification):
        @staticmethod
        def valid(arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            try:
                return arg>0
            except Exception:
                return False
        @staticmethod
        def forgive(arg,level):
            if level==0:
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    pass
    class nonpositive_spec(Specification):
        @staticmethod
        def valid(arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            try:
                return arg<=0
            except Exception:
                return False
        @staticmethod
        def forgive(arg,level):
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
    class negative_spec(Specification):
        @staticmethod
        def valid(arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            try:
                return arg<0
            except Exception:
                return False
        @staticmethod
        def forgive(arg,level):
            if level==0:
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    pass 
    class float_spec(Specification):
        @staticmethod
        def valid(arg):
            return isinstance(arg,builtins.float)
        @staticmethod
        def forgive(arg,level):
            if level==0:
                try:
                    return float(arg)
                except Exception:
                    pass
    class integer_spec(Specification):
        @staticmethod
        def valid(arg):
            return isinstance(arg,builtins.int)
        @staticmethod
        def forgive(arg,level):
            if level==0:
                if isinstance(arg,six.string_types):
                    try:
                        return ast.literal_eval(arg)
                    except Exception:
                        pass
                if isinstance(arg,builtins.float) and arg.is_integer():
                    return int(arg)
                if isinstance(arg,numpy.ndarray) and issubclass(arg.dtype, numbers.Integral):
                    try:
                        return int(arg)
                    except Exception:
                        pass
            if level==1:
                return int(arg)
            
    class ndim_spec(Specification):
        def __init__(self,forgiveness,ndim):
            self.ndim=ndim
            self.forgiveness=forgiveness
        def valid(self,arg):
            return arg.ndim==self.ndim
        def forgive(self,arg,level):
            if level==0:
                t = arg.squeeze()
                while t.ndim<self.ndim:
                    t=numpy.expand_dims(t,axis=-1)
                return t
        
    class shape_spec(Specification):
        def __init__(self,forgiveness,shape):
            self.shape=shape
            self.forgiveness=forgiveness
        def valid(self,arg):
            return arg.shape==self.shape
        def forgive(self,arg,level):
            if level==0:
                return arg.reshape(self.shape)
        
    class function_spec(Specification):
        def __init__(self,forgiveness,spec_string=None):
            if spec_string:
                self.spec_tree=get_specification_tree(spec_string)
            else:
                self.spec_tree = None
            self.forgiveness=forgiveness
        #@staticmethod
        def valid(self,arg):
            return hasattr(arg,'__call__')
        #@staticmethod
        def forgive(self,arg,level):
            if level==0:
                return lambda x:arg
        def final(self,arg):
            if self.spec_tree:
                import swutil.validation_old
                return swutil.validation_old.ValidatedFunction(arg,self.spec_tree)
            else:
                return arg
    
    class dict_spec(Specification):
        @staticmethod
        def valid(arg):
            return isinstance(arg,builtins.dict)
        @staticmethod
        def forgive(arg,level):
            if level==0:
                ret=dict()
                try:
                    for i,v in arg.items():
                        ret[i]=v
                except Exception:
                    pass
            if level==1:
                if isinstance(arg,(builtins.list,builtins.tuple)):
                    d=builtins.dict()
                    for i in range(len(arg)):
                        d[i]=arg[i]
                    return d
    class choice_spec(Specification):
        def __init__(self,forgiveness,*choices):
            self.forgiveness=forgiveness
            self.choices=choices
            self.mod_choices=tuple(self.choices)
        def valid(self,arg):
            return arg in self.choices
        def forgive(self,arg,level):
            if level==0:
                if isinstance(arg,six.string_types):
                    string_choices=[choice for choice in self.choices if isinstance(choice,six.string_types)]
                    if all(s[0].isupper() for s in string_choices) or all(s[0].islower() for s in string_choices):
                        mod_choices=[choice.lower() if hasattr(choice,'lower') else choice for choice in self.choices]
                        if len(set(mod_choices))==len(mod_choices) and arg.lower() in mod_choices:#uniqueness
                            self.mod_choices=mod_choices
                            return self.choices[self.mod_choices.index(arg.lower())]
            if level==1:
                l=0
                if isinstance(arg,six.string_types):
                    string_choices=[choice for choice in self.choices if isinstance(choice,six.string_types)]
                    if all(s[0].isupper() for s in string_choices) or all(s[0].islower() for s in string_choices):
                        arg=arg.lower()
                        mod_choices=[choice.lower() if hasattr(choice,'lower') else choice for choice in self.choices]
                    else:
                        mod_choices=self.choices
                    if len(set(mod_choices))==len(mod_choices):
                        for l in range(max(len(s) for s in string_choices)):
                            l+=1
                            inits=[choice[:l] if isinstance(choice,six.string_types) else choice for choice in mod_choices ] 
                            if len(inits)==len(set(inits)) and arg in inits:
                                return self.choices[inits.index(arg)]
    class set_spec(Specification):
        @staticmethod
        def valid(arg):
            return isinstance(arg,builtins.set)
        @staticmethod
        def forgive(arg,level):
            if level==0:#Convert lists with unique entries
                if not isinstance(arg,six.string_types):
                    temp=set(arg)
                    if len(temp)==len(arg):
                        return temp
            if level==1:#Convert lists with repetitions and single objects
                if not isinstance(arg,six.string_types):
                    try:
                        return set(arg)
                    except Exception:
                        pass
                return {arg}
            
    class list_spec(Specification):
        @staticmethod
        def valid(arg):
            return isinstance(arg,builtins.list)
        @staticmethod
        def forgive(arg,level):
            if level==0:
                if not isinstance(arg,six.string_types):
                    return builtins.list(arg)
            if level==1:
                return [arg]
            
    class tuple_spec(Specification):
        @staticmethod
        def valid(arg):
            return isinstance(arg,builtins.tuple)
        @staticmethod
        def forgive(arg,level):
            if level==0:
                if not isinstance(arg,six.string_types):
                    return builtins.tuple(arg)
            if level==1:
                return (arg,)
                
    class lower_spec(Specification):
        @staticmethod
        def valid(arg):
            return hasattr(arg,'lower') and arg.lower()==arg
        @staticmethod
        def forgive(arg,level):
            if level==1:
                if hasattr(arg,'lower'):
                    return arg.lower()
                
    class upper_spec(Specification):
        @staticmethod
        def valid(arg):
            return hasattr(arg,'upper') and arg.upper()==arg
        @staticmethod
        def forgive(arg,level):
            if level==1:
                if hasattr(arg,'upper'):
                    return arg.upper()

    class in_interval_spec(Specification):
        def __init__(self,forgiveness,a,b,lo=False,ro=False):
            self.a=a
            self.b=b
            self.lo=lo
            self.ro=ro
            self.forgiveness=forgiveness
        def valid(self,arg):
            if not (isinstance(arg,numbers.Number) and type(arg)!=builtins.bool):
                return False
            if self.lo:
                if arg<=self.a:
                    return False
            else:
                if arg<self.a:
                    return False
            if self.ro:
                if arg>=self.b:
                    return False 
            else:
                if arg>self.b:
                    return False
            return True
        def forgive(self,arg,level):
            if level==0:
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    pass 
            if level==1:
                try:
                    if not self.lo:
                        arg=max(arg,self.a)
                    if not self.ro:
                        arg=min(arg,self.b)
                    return arg
                except Exception:
                    pass
    class in_range_spec(in_interval_spec):
        def __init__(self,forgiveness,a,b=None,lo=False,ro=True):
            if b is None:
                b=a
                a=0
            super().__init__(forgiveness,a,b,lo,ro)            
    class any_spec(Specification):
        def __init__(self,forgiveness,description=None):
            self.description=description
        def valid(self,arg):
            return True
    class length_spec(Specification):
        def __init__(self,forgiveness,l_min,l_max=None):
            self.l_min=l_min
            if l_max:
                self.l_max=l_max
            else:
                self.l_max=l_min
            self.forgiveness=forgiveness
        def valid(self,arg):
            return self.l_min<=len(arg)<=self.l_max

    class iterable_spec(Specification):
        def __init__(self,forgiveness):
            self.forgiveness=forgiveness
        def valid(self,arg):
            if not isinstance(arg,six.string_types):
                if not standards.allows_spec(0,'items').valid(arg):
                    try:
                        _=(_ for _ in arg)
                        return True
                    except Exception:
                        pass
        def forgive(self,arg,level):
            if level==1:
                return (arg,)
            
    class allows_spec(Specification):
        def __init__(self,forgiveness,*fn_names):
            self.forgiveness=forgiveness
            self.fn_names=fn_names
        def valid(self,arg):
            try:
                return all(standards.function_spec.valid(getattr(arg,fn_name)) for fn_name in self.fn_names)
            except Exception:
                pass
    class has_spec(Specification):
        def __init__(self,forgiveness,*attr_names):
            self.forgiveness=forgiveness
            self.attr_names=attr_names
        def valid(self,arg):
            return all(hasattr(arg,attr_name) for attr_name in self.attr_names)
    class implements_spec(Specification):
        def __init__(self,forgiveness,*declarations):
            self.forgiveness=forgiveness
            self.declarations=declarations
        def valid(self,arg):
            try:
                return all(find_implementation(arg.__class__, declaration) for declaration in self.declarations)
            except Exception:
                pass
    class class_spec(Specification):
        def __init__(self,forgiveness,cl):
            self.forgiveness=forgiveness
            self.cl=cl
        def valid(self,arg):
            return isinstance(arg,self.cl)
        def forgive(self,arg,level):
            if level==0:
                return self.cl(arg)
    class satisfies_spec(Specification):
        def __init__(self,forgiveness,*checks):
            self.checks=checks
            self.forgiveness=forgiveness
        def valid(self,arg):
            return all(check(arg) for check in self.checks)
        
def void():
        a=numpy.random.random((70,70))*numpy.random.random((70,70))
def test():
        get_specification_tree('integer|list|set|lower|upper|complies*',aux=(SpecificationTree(),))




if __name__=='__main__':
    test()
    #print_runtime(void)()
    
    