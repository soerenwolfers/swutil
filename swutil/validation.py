import six
import ast
import builtins
import numbers
from math import inf
from swutil.decorators import find_implementation, default
from swutil.aux import NotPassed, Passed
import itertools
from _collections import defaultdict
from copy import deepcopy
import inspect
import functools
from inspect import signature
from abc import ABC
from swutil.errors import ValidationError
# TODO: _validate should always raise error instead of returning None

class Specification(ABC):
    def __init__(self, lenience=1):
        if (not isinstance(lenience, builtins.int)) or lenience < 0:
            raise TypeError('`lenience` must be a positive integer')
        self.lenience = lenience
    def __call__(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)
    def _validate(self, arg):  # unforgiving version, only to be used by validate 
        if self.valid(arg):
            return arg
        else:
            raise ValidationError()
    def forgive(self, arg, level):
        return arg
    def __rand__(self, other):
        return All(other, self)
    def __and__(self, other):
        return All(self, other)
    def __ror__(self, other):
        return Or(other, self)
    def __or__(self, other):
        return Or(self, other)
    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return str(self)
    def __format__(self, format_spec):
        return str(self) 
     
def validate(arg, spec):
    if not isinstance(spec, Specification):
        raise TypeError('`{}` must be instance of `Specification`'.format(spec))  
    if isinstance(spec, Specification):
        rejection_reason = '`{}` was rejected by `{}`.'.format(arg, spec)
        rejection_subreason = ''
        attempt = arg
        if not hasattr(spec, 'lenience'):
            spec.lenience = 0
        retry = True
        for level in range(1, spec.lenience + 2):
            if retry:
                try:  # only when change through forgiveness
                    return spec._validate(attempt)
                except Exception as e:
                    if isinstance(e, ValidationError) and len(e.args) > 0 and e.args[0]:
                        rejection_subreason = ' (' + e.args[0] + ')' 
            retry = False
            if level <= spec.lenience:
                temp = None
                try:
                    temp = spec.forgive(arg=attempt, level=level)
                except Exception as e:
                    pass  # Forgiving might fail, it is very hard to predict what happens when you do stuff to things that aren't what you think
                if temp is not None and temp is not attempt:
                    attempt = temp
                    retry = True
        rejection_reason = rejection_reason + rejection_subreason
        raise ValidationError(rejection_reason)  

_NotPassed_substitutes = {}  # {String:lambda:'',Set:lambda:builtins.set(),Tuple:lambda:builtins.tuple(),List:lambda:builtins.list(),Dict:lambda:builtins.dict(),Iterable:lambda:builtins.list()}
def _validate_many(args, specs, defaults, allow_unknowns,
                  unknowns_specification,
                  bool_conditions,
                  passed_conditions): 
        args = deepcopy(args)
        validated_args = builtins.dict() 
        passed_but_not_specified = set(args.keys()) - set(specs.keys())
        if passed_but_not_specified:
            if not allow_unknowns:
                raise ValueError(('Arguments {} were passed but not specified (use ' + 
                     '`allow_unknowns=True` to avoid this error)'.format(passed_but_not_specified)))
            else:
                for arg in passed_but_not_specified:
                    specs[arg] = unknowns_specification
        for arg in specs:
            if not arg in args or NotPassed(args[arg]):
                if arg in defaults:
                    validated_args[arg] = defaults[arg]
                elif specs[arg] in _NotPassed_substitutes:
                    validated_args[arg] = _NotPassed_substitutes[arg]()
                else:
                    validated_args[arg] = NotPassed
            else:
                validated_args[arg] = validate(args[arg], specs[arg])
        if bool_conditions or passed_conditions:
            validated_args = validate(validated_args, Dict(bool_conditions=bool_conditions, passed_conditions=passed_conditions))
        return validated_args

class validate_inputs():  # DecoratorFactory
    def __init__(self, bool_conditions=NotPassed, passed_conditions=NotPassed,
                 allow_unknowns=False, unknowns_specification=NotPassed):
        unknowns_specification = unknowns_specification or Any
        if not (NotPassed(bool_conditions) or isinstance(bool_conditions, six.string_types)):
            raise TypeError('`bool_conditions` must be a string')
        if not (NotPassed(passed_conditions) or isinstance(passed_conditions, six.string_types)):
            raise TypeError('`passed_conditions` must be a string')
        if not isinstance(allow_unknowns, builtins.bool):
            raise TypeError('`allow_unknowns` must be boolean')
        if not isinstance(unknowns_specification, Specification):
            raise TypeError('`unknowns_specification` must be instance of `Specification`')
        self.bool_conditions = bool_conditions
        self.passed_conditions = passed_conditions
        self.allow_unknowns = allow_unknowns
        self.unknowns_specification = unknowns_specification
    def __call__(self, f):
        sig_f = signature(f)
        @functools.wraps(f)
        def new_f(*args, **kwargs):
            check_dict = dict()
            spec_dict = dict()
            args = list(args)
            try_args = True
            defaults = {}
            for name, param in sig_f.parameters.items():
                if param.kind not in [2, 4]:
                    v = NotPassed
                    if param.kind == 3 and args:  # When does this happen?
                        raise ValueError("Parameter `{}` is keyword-only".format(name))
                    if try_args and args:
                        v = args.pop(0)
                    elif name in kwargs:
                        v = kwargs.pop(name)
                    if Passed(v):
                        check_dict[name] = v
                elif param.kind == 2:
                    if args:
                        v = args
                        try_args = False
                        check_dict[name] = v
                elif param.kind == 4:
                    if kwargs:
                        v = kwargs
                        check_dict[name] = v
                if param.annotation is not inspect._empty:
                    spec = validate(param.annotation, Instance(Specification))
                    if param.kind == 4:
                        spec = Dict(key_spec=String, value_spec=spec)
                    elif param.kind == 2:
                        spec = Iterable(value_spec=spec)
                else:
                    spec = Any
                if param.default is not inspect._empty:
                    default = deepcopy(param.default)
                    defaults[name] = default
                spec_dict[name] = spec
            try:
                check_dict = _validate_many(args=check_dict, specs=spec_dict, defaults=defaults,
                    bool_conditions=self.bool_conditions,
                    passed_conditions=self.passed_conditions,
                    allow_unknowns=self.allow_unknowns,
                    unknowns_specification=self.unknowns_specification
                )
            except ValidationError as e:
                msg = e.args[0] if e.args else ''
                raise TypeError(msg) from None
            call_args = []
            call_kwargs = {}
            fill = 'args'
            for name, param in signature(f).parameters.items():
                if not param.kind in [2, 4]:
                    if fill == 'args':
                        call_args.append(check_dict[name])
                    else:
                        call_kwargs[name] = check_dict[name]
                elif param.kind == 2:
                    call_args += check_dict[name]
                    fill = 'kwargs'
                elif param.kind == 4:
                    call_kwargs.update(check_dict[name])
            return f(*call_args, **call_kwargs)
        return new_f

class All(Specification):
    def __init__(self, *specs):
        
        self.specs = specs
        super().__init__()
    def _validate(self, arg):
        for spec in self.specs:
            arg = validate(arg, spec)
        return arg
    def __str__(self):
        return 'All(' + ','.join(str(spec) for spec in self.specs) + ')'

class Or(Specification):
    def __init__(self, *specs):
        if any(not isinstance(spec, Specification) for spec in specs):
            raise TypeError('`specs` must be instances of `Specification`')
        self.specs = specs
        super().__init__()
    def _validate(self, arg):
        for spec in self.specs:
            rejection_reason = ''
            try:
                return validate(arg, spec)
            except Exception as e:
                if isinstance(e, ValidationError) and len(e.args) > 0 and e.args[0]:
                    rejection_reason += +e.args[0]
        raise ValidationError(rejection_reason)
    def __str__(self):
        return 'Or(' + ','.join(str(spec) for spec in self.specs) + ')'
    
class _Bool(Specification):
    @staticmethod
    def valid(arg):
        return isinstance(arg, builtins.bool)
    @staticmethod
    def forgive(arg, level):
        if level == 1:
            if arg in ['True', 'true']:
                return True
            if arg in ['False', 'false']:
                return False
        if level == 2:
            return builtins.bool(arg)
    def __str__(self):
        return 'Bool'
Bool = _Bool()

class _String(Specification):
    @staticmethod
    def valid(arg):
        return isinstance(arg, six.string_types)
    @staticmethod
    def forgive(arg, level):
        if level == 2:
            return builtins.str(arg)
    def __str__(self):
        return 'String'
String = _String()

class _Float(Specification):
    @staticmethod
    def valid(arg):
        return isinstance(arg, builtins.float)
    @staticmethod
    def forgive(arg, level):
        if level == 1:
            try:
                return float(arg)
            except Exception:
                pass
    def __str__(self):
        return 'Float'
Float = _Float()

class _Integer(Specification):
    @staticmethod
    def valid(arg):
        return isinstance(arg, builtins.int)
    @staticmethod
    def forgive(arg, level):
        import numpy
        if level == 1:
            if isinstance(arg, six.string_types):
                try:
                    return ast.literal_eval(arg)
                except Exception:
                    pass
            if isinstance(arg, builtins.float) and arg.is_integer():
                return int(arg)
            if isinstance(arg, numpy.ndarray) and issubclass(arg.dtype, numbers.Integral):
                try:
                    return int(arg)
                except Exception:
                    pass
        if level == 2:
            return int(arg)  
    def __str__(self):
        return 'Integer'
Integer = _Integer()
     
class NDim(Specification):
    def __init__(self, ndim, lenience=1):
        if not isinstance(ndim, builtins.int):
            raise TypeError('`ndim` must be an integer')
        self.ndim = ndim
        super().__init__(lenience)
    def valid(self, arg):
        return arg.ndim == self.ndim
    def forgive(self, arg, level):
        import numpy
        if level == 1:
            t = arg.squeeze()
            while t.ndim < self.ndim:
                t = numpy.expand_dims(t, axis=-1)
            return t
    def __str__(self):
        return 'NDim({})'.format(self.ndim)
    
class _Tuple(Specification):
    def __init__(self, value_spec=NotPassed, lenience=1):
        if not (NotPassed(value_spec) or isinstance(value_spec, Specification)):
            raise TypeError('`value_spec` must be instance of `Specification`')
        self.value_spec = value_spec
        super().__init__(lenience)
    def valid(self, arg):
        return isinstance(arg, builtins.tuple)
    def _validate(self, arg):
        if not self.valid(arg):
            raise ValidationError('Not a tuple')
        if self.value_spec:
            l = [validate(entry, self.value_spec) for entry in arg]
            if any(list_entry != tuple_entry for (list_entry, tuple_entry) in zip(l, arg)):
                return builtins.tuple(l)
            else:
                return arg
        else:
            return arg
    def forgive(self, arg, level):
        if level == 1:
            if not isinstance(arg, six.string_types):
                return builtins.tuple(arg)
        if level == 2:
            return (arg,)
    def __str__(self):
        return 'Tuple{}'.format('({})'.format(self.value_spec) if self.value_spec else '')
Tuple = _Tuple()

class Shape(Specification):
    def __init__(self, shape, lenience=1):
        if not isinstance(shape, (builtins.list, builtins.tuple)):
            raise TypeError('`shape` must be list of integers')
        self.shape = shape
        super().__init__(lenience)
    def valid(self, arg):
        return arg.shape == self.shape
    def forgive(self, arg, level):
        import numpy
        if level == 1:
            if numpy.squeeze(arg).shape == numpy.squeeze(numpy.zeros(self.shape)).shape:
                return arg.reshape(self.shape)
        if level == 2:
            return arg.reshape(self.shape)
    def __str__(self):
        return 'Shape({})'.format(self.shape)


class ValidatedFunction():
    def __init__(self, fn, value_spec):
        self.fn = fn
        self.value_spec = value_spec  
    def __call__(self, *args, **kwargs):
        return validate(self.fn(*args, **kwargs), self.value_spec)

class _Function(Specification):
    def __init__(self, value_spec=NotPassed, lenience=1):
        if not (NotPassed(value_spec) or isinstance(value_spec, Specification)):
            raise TypeError('`value_spec` must be instance of `Specification`')
        if Passed(value_spec):
            self.value_spec = validate(value_spec, Instance(Specification))
        self.value_spec = value_spec
        super().__init__(lenience)
    def forgive(self, arg, level):
        if level == 1:
            if Passed(self.value_spec):
                validate(arg, self.value_spec)
            return lambda x:arg  # Value will also be checked when needed
    def valid(self, arg):
        return hasattr(arg, '__call__')
    def _validate(self, arg):
        if self.valid(arg):
            if Passed(self.value_spec):
                return ValidatedFunction(arg, self.value_spec)
            else:
                return arg
        else:
            raise ValidationError()
    def __str__(self):
        return 'Function{}'.format('({})'.format(self.value_spec) if self.value_spec else '')
Function = _Function()

class _Dict(Specification):
    def __init__(self, key_spec=NotPassed, value_spec=NotPassed, lenience=1, bool_conditions=NotPassed, passed_conditions=NotPassed):
        if not (NotPassed(key_spec) or isinstance(key_spec, Specification)):
            raise TypeError('`key_spec` must be instance of `Specification`')
        if not (NotPassed(value_spec) or isinstance(value_spec, Specification)):
            raise TypeError('`value_spec` must be instance of `Specification`')
        if not (NotPassed(bool_conditions) or isinstance(bool_conditions, six.string_types)):
            raise TypeError('`bool_conditions` must be a string')
        if not (NotPassed(passed_conditions) or isinstance(passed_conditions, six.string_types)):
            raise TypeError('`passed_conditions` must be a string')
        self.key_spec = key_spec
        self.value_spec = value_spec
        self.conditions_dict = {'bool':bool_conditions, 'passed':passed_conditions}
        self.required_results_iterables_and_connectives_dict = {}
        for type in self.conditions_dict:
            if Passed(self.conditions_dict[type]):
                self.required_results_iterables_and_connectives_dict[type] = []
                for condition in [r for r in self.conditions_dict[type].split(' ') if r != '']:
                    required_results_groups, connective = _parse_condition(condition)
                    required_results_groups = [_parse_required_results_group(required_results_group) for required_results_group in required_results_groups]
                    required_results_iterable = list(itertools.product(*(required_results_group.items() for required_results_group in required_results_groups)))
                    self.required_results_iterables_and_connectives_dict[type].append((required_results_iterable, connective))
        super().__init__(lenience)
    def valid(self, arg):
        return isinstance(arg, builtins.dict)
    def _validate(self, arg):
        if not self.valid(arg):
            raise ValidationError('Not a dictionary')
        if Passed(self.value_spec) or Passed(self.key_spec):
            if NotPassed(self.value_spec):
                self.value_spec = Any
            if NotPassed(self.key_spec):
                self.key_spec = Any
            l = [(validate(key, self.key_spec), validate(arg[key], self.value_spec)) for key in arg]
            if any(list_item != dict_item for (list_item, dict_item) in zip(l, arg.items())):
                arg = builtins.dict(l)
        for type in self.required_results_iterables_and_connectives_dict:
            arg = _validate_dict_conditions(arg, self.required_results_iterables_and_connectives_dict[type], type=type)
        return arg 
    def forgive(self, arg, level):
        if level == 1 and not self.valid(arg):
            ret = builtins.dict()
            try:
                for key in arg:
                    ret[key] = arg[key]
                return ret
            except Exception:
                pass
        if level == 2:
            if isinstance(arg, (builtins.list, builtins.tuple)):
                d = builtins.dict()
                for i in range(len(arg)):
                    d[i] = arg[i]
                return d
    def __str__(self):
        init_arguments = (self.key_spec, self.value_spec, self.conditions_dict['bool'], self.conditions_dict['passed'])
        out = 'Dict'
        if any(Passed(init_argument) for init_argument in init_arguments):
            out += '('
            if any(Passed(init_argument) for init_argument in init_arguments[:2]):
                out += 'key_spec={},value_spec={},'.format(*init_arguments[:2])
            if any(Passed(init_argument) for init_argument in init_arguments[2:]):
                out += 'bool_conditions=\'{}\',passed_conditions=\'{}\','.format(*init_arguments[2:])
            out = out[:-1] + ')'
        return out
Dict = _Dict()

class Choice(Specification):
    def __init__(self, *choices, lenience=1):
        self.choices = choices
        self.mod_choices = tuple(self.choices)
        super().__init__(lenience)
    def valid(self, arg):
        return arg in self.choices
    def forgive(self, arg, level):
        if level == 1:  # Forgive capitalization
            if isinstance(arg, six.string_types):
                string_choices = [choice for choice in self.choices if isinstance(choice, six.string_types)]
                # Reasoning: If different first letter capitalization sure capitalization has l meaning, so no forgiving
                # If same first letter capitalization, but pairs like AB vs. Ab occur then of course also no forgiving 
                if all(s[0].isupper() for s in string_choices) or all(s[0].islower() for s in string_choices):
                    mod_choices = [choice.lower() if hasattr(choice, 'lower') else choice for choice in self.choices]  # keep non string choices so that index below works
                    if len(set(mod_choices)) == len(mod_choices) and arg.lower() in mod_choices:
                        self.mod_choices = mod_choices
                        return self.choices[self.mod_choices.index(arg.lower())]
        if level == 2:  # Forgive abbreviations
            l = 0
            if isinstance(arg, six.string_types):
                string_choices = [choice for choice in self.choices if isinstance(choice, six.string_types)]
                if all(s[0].isupper() for s in string_choices) or all(s[0].islower() for s in string_choices):
                    arg = arg.lower()
                    mod_choices = [choice.lower() if hasattr(choice, 'lower') else choice for choice in self.choices]
                else:
                    mod_choices = self.choices
                if len(set(mod_choices)) == len(mod_choices):
                    for l in range(max(len(s) for s in string_choices)):
                        l += 1
                        inits = [choice[:l] if isinstance(choice, six.string_types) else choice for choice in mod_choices ] 
                        if len(inits) == len(set(inits)) and arg in inits:
                            return self.choices[inits.index(arg)]
    def __str__(self):
        return 'Choice(' + ','.join(str(spec) for spec in self.choices) + ')'
    
class _Set(Specification):
    def __init__(self, value_spec=NotPassed, lenience=1):
        if not (NotPassed(value_spec) or isinstance(value_spec, Specification)):
            raise TypeError('`value_spec` must be instance of `Specification`')
        self.value_spec = value_spec
        super().__init__(lenience)
    def valid(self, arg):
        return isinstance(arg, builtins.set)
    def _validate(self, arg):
        if not self.valid(arg):
            raise ValidationError('Not a set')
        if self.value_spec:
            l = [validate(entry, self.value_spec) for entry in arg]
            if any(list_entry != set_entry for (list_entry, set_entry) in zip(l, arg)):
                return builtins.set(l)
            else:
                return arg
        else:
            return arg
    def forgive(self, arg, level):
        if level == 1:  # Convert iterables with unique entries
            if not isinstance(arg, six.string_types):
                temp = builtins.set(arg)
                if len(temp) == len(arg):
                    return temp
        if level == 2:  # Convert lists with repetitions and single objects
            if not isinstance(arg, six.string_types):
                try:
                    return set(arg)
                except Exception:
                    pass
            return {arg}
    def __str__(self):
        return 'Set{}'.format('({})'.format(self.value_spec) if self.value_spec else '')
Set = _Set()

class _List(Specification):
    def __init__(self, value_spec=NotPassed, lenience=1):
        if not (NotPassed(value_spec) or isinstance(value_spec, Specification)):
            raise TypeError('`value_spec` must be instance of `Specification`')
        self.value_spec = value_spec
        super().__init__(lenience)

    def _validate(self, arg):
        if not self.valid(arg):
            raise ValidationError('Not a list')
        if self.value_spec:
            l = [validate(entry, self.value_spec) for entry in arg]
            if any(list_entry != tuple_entry for (list_entry, tuple_entry) in zip(l, arg)):
                return l
            else:
                return arg
        else:
            return arg
    def forgive(self, arg, level):
        if level == 1:
            if not isinstance(arg, six.string_types):
                return builtins.list(arg)
        if level == 2:
            return [arg]
    def valid(self, arg):
        return isinstance(arg, builtins.list)
    def __str__(self):
        return 'List{}'.format('({})'.format(self.value_spec) if self.value_spec else '')
List = _List()

class _Lower(Specification):
    @staticmethod
    def valid(arg):
        return hasattr(arg, 'lower') and arg.lower() == arg
    @staticmethod
    def forgive(arg, level):
        if level == 2:
            if hasattr(arg, 'lower'):
                return arg.lower()
    def __str__(self):
        return 'Lower'
Lower = _Lower()

class _Upper(Specification):
    @staticmethod
    def valid(arg):
        return hasattr(arg, 'upper') and arg.upper() == arg
    @staticmethod
    def forgive(arg, level):
        if level == 2:
            if hasattr(arg, 'upper'):
                return arg.upper()
    def __str__(self):
        return 'Upper'
Upper = _Upper()

class InInterval(Specification):
    def __init__(self, l, r, lo=False, ro=False, lenience=1):
        self.l = l
        self.r = r
        self.lo = lo
        self.ro = ro
        self.lenience = lenience
    def valid(self, arg):
        if not (isinstance(arg, numbers.Number) and type(arg) != builtins.bool):
            return False
        if self.lo:
            if arg <= self.l:
                return False
        else:
            if arg < self.l:
                return False
        if self.ro:
            if arg >= self.r:
                return False 
        else:
            if arg > self.r:
                return False
        return True
    def forgive(self, arg, level):
        if level == 1:
            try:
                return ast.literal_eval(arg)
            except Exception:
                pass 
        if level == 2:
            try:
                if not self.lo:
                    arg = max(arg, self.l)
                if not self.ro:
                    arg = min(arg, self.r)
                return arg
            except Exception:
                pass
    def __str__(self):
        return 'InInterval(l={},r={},lo={},ro={})'.format(self.l, self.r, self.lo, self.ro)

class InRange(InInterval):
    def __init__(self, l, r=NotPassed, lenience=1):
        if NotPassed(r):
            r = l
            l = 0
        super().__init__(lenience=lenience, l=l, r=r, lo=False, ro=True)    

class _Any(Specification):
    def __init__(self, description=NotPassed):
        self.description = description
    def valid(self, arg):
        return True
Any = _Any()

class Length(Specification):
    def __init__(self, l_min, l_max=NotPassed, lenience=1):
        self.l_min = l_min
        if Passed(l_max):
            self.l_max = l_max
        else:
            self.l_max = l_min
        self.lenience = lenience
    def valid(self, arg):
        return self.l_min <= len(arg) <= self.l_max
    def __str__(self):
        return 'Length(l_min={},l_max={})'.format(self.l_min, self.l_max)
    
class _Iterable(Specification):
    def __init__(self, value_spec=NotPassed, lenience=1):
        super().__init__(lenience)
        self.value_spec = value_spec
    def valid(self, arg):
        try:
            validate(arg, Allows('__iter__'))
            if not isinstance(arg, six.string_types):
                return True
        except ValidationError:
            pass
        return False
    def _validate(self, arg):
        if not self.valid(arg) and self.lenience == 2:
            arg = (arg,)
        if self.valid(arg):
            if Passed(self.value_spec):
                return ValidatedIterable(arg, self.value_spec)
            else:
                return arg
        else:
            raise ValidationError()
    def __str__(self):
        return 'Iterable{}'.format('({})'.format(self.value_spec) if self.value_spec else '')
Iterable = _Iterable()     
class ValidatedIterable():
    def __init__(self, iterable, value_specification):
        self.iterable = iterable
        self.value_specification = value_specification
    def __iter__(self):
        checks = zip(self.iterable, itertools.repeat(self.value_specification))
        for v, value_specification in checks:
            yield validate(v, spec=value_specification)

class Allows(Specification):  # Checks if objects have methods (including classmethods and staticmethods)
    def __init__(self, *fn_names, lenience=1):
        self.lenience = lenience
        self.fn_names = fn_names
    def valid(self, arg):
        try:
            return all(Function.valid(getattr(arg, fn_name)) for fn_name in self.fn_names)
        except Exception:
            pass
    def __str__(self):
        return 'Allows(' + ','.join(fn_name for fn_name in self.fn_names) + ')'

class Has(Specification):
    def __init__(self, *attr_names, lenience=1):
        self.lenience = lenience
        self.attr_names = attr_names
    def valid(self, arg):
        return all(hasattr(arg, attr_name) for attr_name in self.attr_names)
    def __str__(self):
        return 'Has(' + ','.join(attr_name for attr_name in self.attr_names) + ')'

class Implements(Specification):
    def __init__(self, *declarations, lenience=1):
        if any(not isinstance(declaration, default) for declaration in declarations):
            fail = [declaration for declaration in declarations if not isinstance(declaration, default)]
            raise ValueError('{} {} not of type `declaration`'.format(fail, 'is' if len(fail) == 1 else 'are'))
        self.lenience = lenience
        self.declarations = declarations
    def valid(self, arg):
        try:
            return all(find_implementation(arg.__class__, declaration) for declaration in self.declarations)
        except Exception:
            pass
    def __str__(self):
        return 'Implements(' + ','.join(str(declaration) for declaration in self.declarations) + ')'
    
class Instance(Specification):
    def __init__(self, cl, lenience=1):
        self.lenience = lenience
        self.cl = cl
    def valid(self, arg):
        return isinstance(arg, self.cl)
    def forgive(self, arg, level):
        if level == 1:
            return self.cl(arg)
    
    def __str__(self):
        return 'Instance({})'.format(self.cl.__qualname__)
    
class Satisfies(Specification):
    def __init__(self, *checks, lenience=1):
        self.checks = checks
        self.lenience = lenience
    def valid(self, arg):
        return all(check(arg) for check in self.checks)
    def __str__(self):
        return 'Satisfies(' + ','.join(str(check) for check in self.checks) + ')'
    
class _Nonnegative(InInterval):
    def __init__(self, lenience=1):
        super().__init__(lenience=lenience, l=0, r=inf, lo=False, ro=False) 
Nonnegative = _Nonnegative()

class _Positive(InInterval):
    def __init__(self, lenience=1):
        super().__init__(lenience=lenience, l=0, r=inf, lo=True, ro=False) 
Positive = _Positive()

class _Nonpositive(InInterval):
    def __init__(self, lenience=1):
        super().__init__(lenience=lenience, l=-inf, r=0, lo=False, ro=False) 
Nonpositive = _Nonpositive()

class _Negative(InInterval):
    def __init__(self, lenience=1):
        super().__init__(lenience=lenience, l=-inf, r=0, lo=False, ro=True) 
Negative = _Negative()

def _parse_condition(req):
    seps = ['|', '^', '>', '-']
    possible_types = [sep for sep in seps if sep in req]
    if len(possible_types) > 1:
        raise ValueError('Each condition can only contain one of the following connectives: {}. '.format(seps) + 
                            'Multiple requirements must be separated by a space character')
    if '|' in req:
        args, req_type = req.split('|'), 'or'
    elif '^' in req:
        args, req_type = req.split('^'), 'xor'
    elif '-' in req:
        args, req_type = req.split('-'), 'iff'
    elif '>' in req:
        if req.count('>') > 1:
            raise ValueError('The implication, `>`, can only occur once in each condition. Multiple conditions must be separated by a space character')
        args, req_type = req.split('>', 1), 'implies'
    else:
        args, req_type = (req,), 'require'
    return args, req_type

def _parse_required_results_group(subject_group):
    if not isinstance(subject_group, six.string_types):
        raise TypeError('subject_group must be a string')
    if subject_group.count('(') > 1 or subject_group.count(')') > 1:
        raise ValueError('Groups of parameters cannot be nested')
    subject = subject_group.strip('( )')
    single_results = subject.split(',')
    parameters_and_results = {single_result.replace('!', ''):bool((1 + single_result.count('!')) % 2) for single_result in single_results}
    if not all(parameter.isidentifier() for parameter in parameters_and_results):
        raise ValueError('Malformed dictionary condition: {}.'.format(subject_group) + 
                         (' Use only the connectives `^`,`|` and `->`' if '&' in subject_group else ''))
    return parameters_and_results

def _validate_dict_conditions(args:Dict, required_results_iterables_and_connectives, type='passed'):
    if not isinstance(args, builtins.dict):
        raise TypeError('`args` must be a dict')
    if not isinstance(required_results_iterables_and_connectives, (builtins.list, builtins.tuple)):
        raise TypeError('`required_results_iterables_and_connectives` must be a list')
    if not type in ('passed', 'bool'):
        raise TypeError('`type` must be either "parsed" or "boolean"')
    args_copy = defaultdict(lambda: NotPassed)
    args_copy.update(args)
    if type == 'passed':
        evaluation = lambda x: Passed(x)
    elif type == 'bool':
        evaluation = lambda x: bool(x)
    condition_string = lambda required_result: type + '(' + required_result[0] + ')' + ('=' if required_result[1] else '!') + '=True'
    for (required_results_iterable, connective) in required_results_iterables_and_connectives:
        if connective == 'or':
            for required_results in required_results_iterable:
                if all(required_result[1] != evaluation(args_copy[required_result[0]]) for required_result in required_results):
                    raise ValidationError('At least one of [{}] must hold'.format(','.join(condition_string(required_result) for required_result in required_results)))
        elif connective == 'implies':
            for (left, right) in required_results_iterable:
                if left[1] == evaluation(args_copy[left[0]]) and right[1] != evaluation(args_copy[right[0]]):
                    raise ValidationError('If  {}, then {} hold as well as well'.format(condition_string(left), condition_string(right)))
        elif connective == 'xor':
            for required_results in required_results_iterable:
                matches = [required_result[1] == evaluation(args_copy[required_result[0]]) for required_result in required_results]
                if matches.count(True) != 1:
                    raise ValidationError('Exactly one of [{}] must hold'.format(','.join(condition_string(required_result) for required_result in required_results)))
        elif connective == 'require':
            for required_results in required_results_iterable:
                required_result = required_results[0]  # Only one subject
                if required_result[1] != evaluation(args_copy[required_result[0]]):
                    raise ValidationError('{} must hold'.format(condition_string(required_result)))
        elif connective == 'iff':
            for required_results in required_results_iterable:
                matches = [required_result[1] == evaluation(args_copy[required_result[0]]) for required_result in required_results]
                if matches.count(True) not in (0, len(required_results)):
                    raise ValidationError('All or none of [{}] must hold'.format(','.join(condition_string(required_result) for required_result in required_results))) 
    return args

# def validate_kwargs(specs,requirements=None,allow_unknowns=False,):
#    args = sys._getframe(1).f_locals['kwargs']
#    validated_args = validate_args(args=args,allow_unknowns=allow_unknowns,specs=specs,requirements=requirements)
#    args.clear()
#    args.update(validated_args)

