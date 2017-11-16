import unittest
from swutil.validation import validate, Nonpositive, ValidationError,\
    Positive, Or, Negative, Function, Integer, All, Bool, String, Float, NDim,\
    Shape, Dict, Choice, Set, List, Tuple, Lower, Upper, Nonnegative, InInterval,\
    InRange, Any, Length, Iterable, Allows, Has, Satisfies, Instance, Implements,\
    validate_inputs
from math import inf
import os
from swutil.decorators import declaration, implementation, print_runtime
from swutil.aux import NotPassed

@declaration
def foo():
    pass 
@validate_inputs('a|b a>c')
def bar(a:Integer,b=1,c=0):
    return a,b,c
@validate_inputs(passed_conditions='a^c')
def bar2(a:Integer,b:String,c):
    return a,b,c
class TestSpecifications(unittest.TestCase):
    def test_validate_inputs(self):
        self.assertEqual(bar(b=1),(NotPassed,1,0))
        with self.assertRaises(TypeError):
            bar(a=1)
        self.assertEqual(bar(a=1,c=2),(1,1,2))
        self.assertEqual(bar2(a=1),(1,NotPassed,NotPassed))
        with self.assertRaises(TypeError):
            bar(a=1,c=0)
        with self.assertRaises(TypeError):
            bar(a='asd')
    def test_Nonpositive(self):
        self.assertEqual(validate(0,Nonpositive),0)
        self.assertEqual(validate(1,Nonpositive(lenience=2)),0)
        with self.assertRaises(ValidationError):
            validate(1,Nonpositive)
    def test_Positive(self):
        self.assertEqual(validate(1,Positive),1)
        with self.assertRaises(ValidationError):
            validate(0,Positive)
    def test_Or(self):
        self.assertEqual(validate(1,Or(Positive,Negative)),1)
        self.assertEqual(validate(1,Positive|Negative),1)
        with self.assertRaises(ValidationError):
            validate("best",Or(Positive,Negative))
        with self.assertRaises(ValidationError):
            validate(0,Positive|Negative)
    def test_Function(self):
        with self.assertRaises(ValidationError):
            validate('mest',Function(lenience=0))
        self.assertEqual(validate('mest',Function)(1),'mest')   
        a=lambda x: 1
        b=validate(a,Function(value_spec=Integer))
        self.assertEqual(b(2),1)
        c=validate(a,Function(value_spec=Negative)) 
        with self.assertRaises(ValidationError):
            c(1)
    def test_All(self):
        self.assertEqual(validate(1,Positive&Integer),1)
        self.assertEqual(validate(2,All(Positive,Integer)),2)
        with self.assertRaises(ValidationError):
            validate(1.4,Positive&Integer)
        with self.assertRaises(ValidationError):
            validate(-1,Positive&Integer)
    def test_Bool(self):
        self.assertEqual(validate('true',Bool),True)
        self.assertEqual(validate('False',Bool),False)
        with self.assertRaises(ValidationError):
            validate('b',Bool)
    def test_String(self):
        self.assertEqual(validate('true',String),'true')
        with self.assertRaises(ValidationError):
            validate(1,String)
    def test_Float(self):
        class tmp(object):
            pass
        self.assertEqual(validate(1.4,Float),1.4)
        self.assertEqual(validate('1.4',Float),1.4)
        with self.assertRaises(ValidationError):
            validate(tmp(),Float)
    def test_Integer(self):
        self.assertEqual(validate(1,Integer),1)
        self.assertEqual(validate('1',Integer),1)
        self.assertEqual(validate(2.1,Integer(lenience=2)),2)
        with self.assertRaises(ValidationError):
            validate(1.5,Integer)
        with self.assertRaises(ValidationError):
            validate('1.5',Integer)
    def test_NDim(self):
        import numpy as np
        self.assertEqual(validate(np.zeros(1),NDim(1)),np.zeros(1))
        self.assertEqual(validate(np.zeros(1),NDim(2)),np.zeros((1,1)))
        self.assertEqual(validate(np.zeros((1,1)),NDim(1)),np.zeros(1))
        with self.assertRaises(ValidationError):
            validate(np.zeros((3,2)),NDim(1))
    def test_Shape(self):
        import numpy as np
        self.assertEqual(validate(np.zeros(1),Shape((1,))),np.zeros(1))
        self.assertEqual(np.array_equal(validate(np.zeros((1,2)),Shape((1,2))),np.zeros((1,2))),True)
        self.assertEqual(np.array_equal(validate(np.zeros((2,3)),Shape((3,2),lenience=2)),np.zeros((3,2))),True)
        with self.assertRaises(ValidationError):
            validate(np.zeros((3,2)),Shape((2,3)))
    def test_Dict(self):
        self.assertEqual(validate({},Dict),{})
        self.assertEqual(validate((1,2),Dict(lenience=2)),{0:1,1:2})
        self.assertEqual(validate({'1':2},Dict(key_spec=Integer)),{1:2})
        self.assertEqual(validate({'1':2},Dict(key_spec=Integer,value_spec=String(lenience=2))),{1:'2'})
        d={'best':1,'cest':2,'test':0}
        self.assertEqual(validate(d,Dict(bool_conditions='best !test best|test best cest (test,!cest)>cest !cest>test')),d)
        self.assertEqual(validate(d,Dict(bool_conditions='best-cest')),d)
        with self.assertRaises(ValidationError):
            validate(d,Dict(bool_conditions='best^cest'))
        with self.assertRaises(ValidationError):
            validate(d,Dict(bool_conditions='best>test'))
        with self.assertRaises(ValidationError):
            validate(d,Dict(bool_conditions='test|test'))
        with self.assertRaises(ValidationError):
            validate(1,Dict)
        with self.assertRaises(ValidationError):
            validate({2:'best'},Dict(value_spec=Integer))
    def test_Choice(self):
        self.assertEqual(validate('best',Choice('test','mest','best')),'best')
        self.assertEqual(validate('Best',Choice('test','mest','best')),'best')
        self.assertEqual(validate('b',Choice('test','mest','best',lenience=2)),'best')
        with self.assertRaises(ValidationError):
            validate('Best',Choice('test','mest','best',lenience=0))
        with self.assertRaises(ValidationError):
            validate('b',Choice('test','best','brest',lenience=2))
        with self.assertRaises(ValidationError):
            validate('best',Choice('test','mest','Best'))
        with self.assertRaises(ValidationError):
            validate('best',Choice('test','tEst','Best'))
    def test_Set(self):
        self.assertEqual(validate({'best'},Set),{'best'})
        self.assertEqual(validate([1,2],Set),{1,2})
        self.assertEqual(validate([1,1],Set(lenience=2)),{1})
        self.assertEqual(validate('best',Set(lenience=2)),{'best'})
        with self.assertRaises(ValidationError):
            validate([1,1],Set)
        with self.assertRaises(ValidationError):
            validate((1,),Set(lenience=0)) 
    def test_List(self):
        self.assertEqual(validate([1],List),[1])
        self.assertEqual(validate((1,),List),[1])
        self.assertEqual(validate(('1',),List(value_spec=Integer)),[1])
        self.assertEqual(validate((str(i) for i in range(3)),List(value_spec=Integer)),[0,1,2])
        with self.assertRaises(ValidationError):
            validate(('1',),List(value_spec=List))
        with self.assertRaises(ValidationError):
            validate((1,),List(value_spec=Integer,lenience=0))
    def test_Tuple(self):
        self.assertEqual(validate((1,),Tuple),(1,))
        self.assertEqual(validate([1],Tuple),(1,))
        self.assertEqual(validate(['1',],Tuple(value_spec=Integer)),(1,))
        with self.assertRaises(ValidationError):
            validate(['1',],Tuple(value_spec=Tuple))
        with self.assertRaises(ValidationError):
            validate([1,],Tuple(value_spec=Integer,lenience=0))
    def test_Lower(self):
        self.assertEqual(validate('lower',Lower),'lower')
        self.assertEqual(validate('LOwER',Lower(lenience=2)),'lower')
        with self.assertRaises(ValidationError):
            validate(1,Lower)
        with self.assertRaises(ValidationError):
            validate('lOwer',Lower)
    def test_Upper(self):
        self.assertEqual(validate('UPPER',Upper),'UPPER')
        self.assertEqual(validate('UPpeR',Upper(lenience=2)),'UPPER')
        with self.assertRaises(ValidationError):
            validate(1,Upper)
        with self.assertRaises(ValidationError):
            validate('UPper',Upper)
    def test_InInterval(self):
        self.assertEqual(validate(0,InInterval(l=-1,r=1)),0)
        self.assertEqual(validate(inf,InInterval(l=0,r=inf)),inf)
        self.assertEqual(validate(-1,InInterval(l=0,r=2,lenience=2)),0)
        with self.assertRaises(ValidationError):
            validate(-1,InInterval(l=1,r=2))
        with self.assertRaises(ValidationError):
            validate(-1,InInterval(l=-1,r=2,lo=True))   
    def test_InRange(self):
        self.assertEqual(validate(0,InRange(1)),0)
        self.assertEqual(validate(2,InRange(inf)),2)
        self.assertEqual(validate(-2,InRange(-2,3)),-2)
        self.assertEqual(validate(-1,InRange(l=0,r=2,lenience=2)),0)
        with self.assertRaises(ValidationError):
            validate(2,InRange(2))
        with self.assertRaises(ValidationError):
            validate(-1,InRange(3))
    def test_Any(self):
        for arg in (unittest,{},str,unittest.main,lambda x:lambda:2,None,1,'best'):
            self.assertEqual(validate(arg,Any),arg)
    def test_Length(self):
        import numpy as np
        self.assertEqual(validate((1,2),Length(2)),(1,2))
        self.assertEqual(validate([2,3],Length(2)),[2,3])
        self.assertEqual(np.array_equal(validate(np.zeros((3,2)),Length(3)),np.zeros((3,2))),True)
        self.assertEqual(validate({1:2,3:4},Length(2)),{1:2,3:4})
        self.assertEqual(validate({},Length(0)),{})
        with self.assertRaises(ValidationError):
            validate({},Length(1))
        with self.assertRaises(ValidationError):
            validate({1},Length(0))
    def test_Iterable(self):
        self.assertEqual(validate({1,2},Iterable),{1,2})
        self.assertEqual(validate((),Iterable),())
        with self.assertRaises(ValidationError):
            validate('asdf',Iterable)
        with self.assertRaises(ValidationError):
            validate(lambda:1, Iterable)
    def test_Allows(self):
        self.assertEqual(validate({1,2},Allows('clear')),{1,2})
        self.assertEqual(validate(TestSpecifications,Allows('__init__')),TestSpecifications)
        with self.assertRaises(ValidationError):
            validate({1,2},Allows('__clear__'))
        with self.assertRaises(ValidationError):
            validate(Any,Allows(''))
        with self.assertRaises(ValidationError):
            validate(os,Allows('__builtins__'))
    def test_Has(self):
        self.assertEqual(validate({1,2},Allows('clear')),{1,2})
        self.assertEqual(validate(TestSpecifications,Has('__init__')),TestSpecifications)
        with self.assertRaises(ValidationError):
            validate({1,2},Has(''))
        with self.assertRaises(ValidationError):
            validate(Any,Has(''))
        self.assertEqual(validate(os,Has('__builtins__')),os)
    def test_Satisfies(self):
        self.assertEqual(validate({1,2},Satisfies(lambda x:len(x)==2)),{1,2})
        with self.assertRaises(ValidationError):
            validate('Best',Satisfies(lambda x: x.is_lower()))
    def test_Instance(self):
        class A():
            pass
        class B():
            pass
        class C(object):
            def __init__(self,arg):
                if arg>0:
                    raise ValueError("arg must be nonpositive")
            def __eq__(self,other):
                if other.__class__==C:
                    return True
                else:
                    return False
        self.assertEqual(validate('best',Instance(str)),'best')
        a=A()
        self.assertEqual(validate(a,Instance(A)),a)
        with self.assertRaises(ValidationError):
            validate('best',Instance(int))
        with self.assertRaises(ValidationError):
            validate(a,Instance(B))
        c=C(-1)
        self.assertEqual(validate(c,Instance(C)),c)
        self.assertEqual(validate(-1,Instance(C)),c)
        with self.assertRaises(ValidationError):
            validate(1,Instance(C))   
    def test_Implements(self):    
        class A(object):
            @implementation(foo)
            def best(self):
                print('hello')
        class B(object):
            def best(self):
                print('mello')
        a=A()
        self.assertEqual(validate(a,Implements(foo)),a)
        with self.assertRaises(ValueError):
            validate(a,Implements(unittest.main))
        b=B()
        with self.assertRaises(ValidationError):
            validate(b,Implements(foo))
    def test_Concat(self):
        self.assertEqual(validate([[[1]]],List(List(List(value_spec=Integer)))),[[[1]]])  
        with self.assertRaises(ValidationError):
            validate([[1]],List(List(String)))    
    def test_Nonnegative(self):
        self.assertEqual(validate(0,Nonnegative),0)
        with self.assertRaises(ValidationError):
            validate(-1,Nonnegative)
    def test_Negative(self):
        self.assertEqual(validate(-1,Negative),-1)
        with self.assertRaises(ValidationError):
            validate(0,Negative)
            
if __name__=="__main__":
    print_runtime(unittest.main)(exit=False)
    #suite=unittest.TestLoader().loadTestsFromName(name='test_specifications.TestSpecifications.test_Choice')
    #unittest.TextTestRunner().run(suite)