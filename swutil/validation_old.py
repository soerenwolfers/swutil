import builtins
import six
import sys
import argparse
from collections import defaultdict
from swutil.specifications import Specification, OrSpecification,\
    get_specification_tree, SpecificationTree, SpecificationLeaf,\
    OrSpecificationTree, standards
from swutil import specifications
from swutil.aux import NotPassed
from swutil.decorators import print_runtime
from copy import deepcopy
import warnings
import inspect
import functools
from inspect import signature
import itertools
from swutil.collections import zip_equal
import collections
from matplotlib.cm import spec
#Define errors globally and raise them by named reference, to allow specific handling
#
class validated(object):
    def __init__(self,requirements=None):
        self.requirements=requirements
    def __call__(self,f):
        #print(f.__annotations__)
        sig_f=signature(f)
        @functools.wraps(f)
        def new_f(*args,**kwargs):
            check_dict=dict()
            spec_dict=dict()
            args=list(args)
            try_args=True
            defaults={}
            for name,param in sig_f.parameters.items():
                if param.kind not in [2,4]:
                    v=NotPassed
                    if param.kind == 3 and args:
                        raise ValueError("Parameter {} is keyword-only".format(name))
                    if try_args and args:
                        v=args.pop(0)
                    elif name in kwargs:
                        v=kwargs.pop(name)
                    if v is not NotPassed:
                        check_dict[name]=v
                elif param.kind==2:
                    if args:
                        v=args
                        try_args=False
                        check_dict[name]=v
                elif param.kind==4:
                    if kwargs:
                        v=kwargs
                        check_dict[name]=v
                aux=[]
                if param.annotation is not inspect._empty:
                    if isinstance (param.annotation,(builtins.tuple,builtins.list)):
                        spec_string=param.annotation[0]
                        aux=list(param.annotation[1:])
                    else:
                        if standards.string_spec(1).valid(param.annotation):
                            spec_string=param.annotation
                            if param.kind==4:
                                spec_string='{string:'+spec_string+'}'
                            elif param.kind==2:
                                spec_string='{'+spec_string+'}'
                        else:
                            spec_string=param.annotation
                else:
                    spec_string='any'
                if param.default is not inspect._empty:
                    default=deepcopy(param.default)
                    defaults[name]=default
                spec_dict[name]=[spec_string]+aux
            validate_args(__check_dict=check_dict,__requirements=self.requirements,__unknowns=False,**spec_dict)
            call_args=[]
            call_kwargs={}
            fill='args'
            for name,param in signature(f).parameters.items():
                if not param.kind in [2,4]:
                    if fill=='args':
                        call_args.append(check_dict[name])
                    else:
                        call_kwargs[name]=check_dict[name]
                elif param.kind ==2:
                    call_args+=check_dict[name]
                    fill='kwargs'
                elif param.kind==4:
                    call_kwargs.update(check_dict[name])
            return f(*call_args,**call_kwargs)
        #print(f.__annotations__)
        return new_f
            
def validate_kwargs(__requirements=None,__unknowns=False,**specs):
    args=sys._getframe(1).f_locals['kwargs']
    return validate_args(args,__requirements,__unknowns=__unknowns,**specs)

def _parse_requirements(req):
    seps= ['|','^','>']
    possible_types=[sep for sep in seps if sep in req]
    if len(possible_types)>1:
        raise ValueError('Each requirement can only contain one of the following conjunctions: {}. '.format(seps)+
                            'Multiple requirements must be separated by a space character')
    if '|' in req:
        args,req_type= req.split('|'),'or'
    elif '^' in req:
        args,req_type =req.split('^'),'xor'
    elif '->' in req:
        args,req_type = req.split('->',1),'implies'
        if args[0][0]=='[' and args[0][-1]==']':
            args[0]=args[0][1:-1].split(',')
    else:
        args,req_type = (req,),'require'
    return args,req_type


_none_substitutes={'string':lambda:'','set':lambda:builtins.set(),'tuple':lambda:builtins.tuple(),'list':lambda:builtins.list(),'dict':lambda:builtins.dict(),'iterable':lambda:builtins.list()}
def validate_args(__check_dict,__requirements=None,__defaults=None,__unknowns=False,**specs): 
        check_dict=__check_dict
        __defaults = __defaults or {}
        assert(specifications.standards.dict_spec(0).valid(check_dict))
        if __unknowns is not False:
            if __unknowns is True:
                __unknowns='' #see line below
            temp2=defaultdict(lambda: __unknowns)#__unknowns functions as default specification of unspecified arguments
            temp2.update(specs)
            specs=temp2
        __requirements=__requirements or ''
        assert(specifications.standards.string_spec(0).valid(__requirements))
        out=builtins.dict() 
        require_all=False
        for requirement in [r for r in __requirements.split(' ') if r!='']:
            req_args,requirement_type=_parse_requirements(requirement)
            if requirement_type=='or':
                if all(not req_arg in check_dict for req_arg in req_args):
                    raise ValueError('At least one of the arguments {} must be passed'.format(req_args))
                for req_arg in req_args:
                    if req_arg not in check_dict:
                        #del specs[req_arg]
                        out[req_arg]=NotPassed
            elif requirement_type=='implies':
                left,right=req_args
                for l in left:
                    if not (l not in check_dict or right in check_dict):
                        raise ValueError('If argument \'{}\' is passed, then \'{}\' must be passed as well'.format(l,right))
                    if l not in check_dict:
                        #del specs[left]
                        out[l]=NotPassed
            elif requirement_type=='xor':
                not_in_args=[req_arg for req_arg in req_args if req_arg not in check_dict]
                if not len(not_in_args)==len(req_args)-1:
                    raise ValueError('Exactly one of the arguments {} must be passed'.format(req_args)) 
                for req_arg in not_in_args:
                    if req_arg not in check_dict:
                        #del specs[left]
                        out[req_arg]=NotPassed
            elif requirement_type=='require':
                req_arg,=req_args
                if req_arg=='*':
                    require_all=True
                elif not req_arg in check_dict:
                    raise ValueError('Argument \'{}\' must be passed'.format(req_arg))
        if require_all and any(arg not in check_dict for arg in specs):
            raise ValueError('Arguments {} were specified but not passed'.format(set(specs.keys())-set(check_dict.keys())))
        if any(arg not in specs for arg in check_dict):
            if __unknowns is False:
                raise ValueError('Arguments {} were passed but not specified (use __unknowns=True to avoid this error)'.format(set(check_dict.keys())-set(specs.keys())))
            else:
                for arg in check_dict:
                    if arg not in specs:
                        specs[arg]=specs[arg]#specs is default dict->uses default specification
        out.update(check_dict)
        for arg in specs:
            if isinstance(specs[arg],(builtins.tuple,builtins.list)):
                aux=tuple(specs[arg][1:])
                spec=specs[arg][0]
            else:
                spec=specs[arg]
                aux=tuple()
            spec_tree=get_specification_tree(spec,aux)
            if not arg in out or out[arg] is NotPassed:
                if arg in __defaults:
                    out[arg]=__defaults[arg]
                else:
                    try:#Try to find 
                        final_leaf=spec_tree.leaves[-1]
                        while isinstance(final_leaf,SpecificationTree):
                            final_leaf=final_leaf.leaves[-1]
                        if isinstance(final_leaf,SpecificationLeaf) and isinstance(final_leaf.specification,Specification):
                            name=final_leaf.name
                            cl=name.split('-')[-1].split('*')[0]
                            cl,*rest=cl.split('(',1)
                            if cl in _none_substitutes:
                                out[arg]=_none_substitutes[cl]()
                            elif cl == 'allows' and rest and rest[0]=='"items")':
                                out[arg]=dict()
                    except Exception:
                        pass
                if not arg in out:
                    out[arg]=NotPassed
            if out[arg] is not NotPassed:
                try:
                    out[arg]=validate(out[arg],spec_tree)
                except ValidationError as e:
                    msg= 'Invalid argument for parameter \'{}\': '.format(arg)
                    msg += '{} was rejected by \'{}\''.format(out[arg].__repr__(),spec)
                    if len(e.args)>0 and e.args[0] not in msg:
                        msg+=' ('+e.args[0]+')'
                    raise ValidationError(msg) from None
                except ValueError as e:
                    msg= 'Invalid argument for parameter \'{}\': '.format(arg)
                    msg += '{} was rejected by \'{}\''.format(out[arg].__repr__(),spec)
                    if len(e.args)>0 and e.args[0] not in msg:
                        msg+=' ('+e.args[0]+')'
                    raise ValidationError(msg) from None
            else:
                out[arg]=NotPassed
        check_dict.clear()
        check_dict.update(out)
        r=argparse.Namespace()
        for name in out:
            setattr(r, name, out[name])
        return r

def validate(value,spec,aux=None):
    if isinstance(spec,six.string_types):
        aux=aux or tuple()
        spec=get_specification_tree(spec,aux,False)
    if isinstance(spec,OrSpecificationTree):
        for leaf in spec.ors:
            try:
                return validate(value,spec=leaf)
            except Exception:
                pass
        raise ValidationError('Invalid value {}'.format(value))
    elif isinstance(spec,SpecificationTree):
        sub_specs=False
        for leaf in spec.leaves[::-1]:
            if leaf.value_specification:
                sub_specs=True
            value=validate(value,spec=leaf)
        if sub_specs:
            for leaf in spec.leaves[::-1]:
                if not leaf.value_specification:
                    value=validate(value,spec=leaf)
        if spec.value_specification or spec.value_specifications:
            if spec.key_specification:
                dict_value=dict()
                changed=False
                for i,v in value.items():
                    ni=validate(i,spec=spec.key_specification)
                    nv=validate(v,spec=spec.value_specification)
                    dict_value[ni]=nv
                    if ni!=i or nv!=v:
                        changed=True
                if changed:
                    value=dict_value
            else:
                #RETURN something that can be called if value can be called, can be index if value can be indexed, can be iterated if value can be iterated
                value=ValidatedIterable(value, spec.value_specifications, spec.value_specification)
                #list_value=list()
                #changed=False
                #if not spec.value_specifications:#In this case, both list type and function type makes sense
                #    checks=zip(value,itertools.repeat(spec.value_specification))
                #else:# In this case, callable doesn't make sense. 
                #    checks=zip_equal(value,spec.value_specifications)
                #for v,value_specification in checks:
                #    nv=validate(v,spec=value_specification)
                #    list_value.append(nv)
                #try:
                #    i=0
                #    for v in value:
                #        if v!=list_value[i]:
                #            raise Exception
                #        i+=1
                #    if i!=len(list_value):
                #        raise Exception
                #except Exception:
                #    warnings.warn('Converted argument of type {} into list'.format(type(value)))
                #    value=list_value
        return value
    elif isinstance(spec,SpecificationLeaf):
        try:
            return validate(value,spec=spec.specification)
        except ValidationError:
            msg='{} was rejected by \'{}\''.format(value.__repr__(),spec.name)
            raise ValidationError(msg) from None
    elif isinstance(spec,OrSpecification):#TODO MOVE THIS AND CASE BELOW TO SPECIFICATION SO IT CAN BE USED WITHOUT DEPENDENCIES
        try:
            if spec.valid(value):
                return value  
        except Exception:
            pass
        for outer_level in range(spec.forgiveness):
            for specification in spec.specifications:
                inner_attempt=value
                if outer_level<=specification.forgiveness:
                    for level in range(outer_level):
                        try:
                            temp=specification.forgive(arg=inner_attempt,level=level)
                            if spec.valid(temp):
                                return temp
                            elif temp is not None:
                                inner_attempt=temp
                        except Exception:
                            pass
    elif isinstance(spec,Specification):
        attempt=value
        #try:
        #    if spec.valid(attempt):
        #        return attempt
        #except Exception:
        #    pass
        if not hasattr(spec,'forgiveness'):
            spec.forgiveness=0
        for level in range(spec.forgiveness+1):
            try:
                if spec.valid(attempt):
                    if hasattr(spec,'final'):
                        attempt=spec.final(attempt)
                    return attempt
                temp=spec.forgive(arg=attempt,level=level)
                if temp is not None:
                    attempt=temp
            except Exception:
                pass
    elif hasattr(spec,'__call__'):
        try:
            return spec(value)
        except Exception:
            raise ValidationError('{} was rejected by {}'.format(value,spec))
    raise ValidationError('Did not know what to do with {}'.format(spec))
    
class ValidationError(Exception):
    pass

class ValidatedFunction(object):
    def __init__(self,fn,value_specification):
        self.fn=fn
        self.value_specification=value_specification
        
    def __call__(self,*args,**kwargs):
        return validate(self.fn(*args,**kwargs),self.value_specification)
        
class ValidatedIterable(object):
    def __init__(self,iterable,value_specifications,value_specification):
        self.iterable=iterable
        self.value_specifications=value_specifications
        self.value_specification=value_specification
        
    def __iter__(self):
        if not self.value_specifications:
            checks=zip(self.iterable,itertools.repeat(self.value_specification))
        else:# 
            checks=zip_equal(self.iterable,self.value_specifications)
        for v,value_specification in checks:
            yield validate(v,spec=value_specification)
  
    #def __call__(self,x):
    #    if self.value_specifications:
    #        raise AttributeError('Cannot call ValidatedList')
    #    return validate(self.iterable(x),self.value_specification)
        

@print_runtime
def test(**kwargs):
    for i in range(1):
        args=validate_kwargs(a='{integer,integer,integer}',c=('{v-class*:class*}',int,str),__defaults={'b':1,'c':{1:'validate_args'}})
    print(kwargs)
        #kwargs=deepcopy(kwargs_old)
    #print(kwargs)
    #print(type(validate([1,2],'string')))
#def f(a:FUNCTION[STRING],b:ALL[OR(NONNEGATIVE,BOOL),INTEGER,CLASS[THIS]],c:ITERABLE[this],d:DICT[a,b]):
@validated()
def f(a:('function*','string'),c:'integer'):  
    print(a,c) 
    print(a(1))
    pass
if __name__=='__main__':
    f(lambda x: 'ada','1')
    
    #print(_get_tuple('(1,{"asd":2},2)'))