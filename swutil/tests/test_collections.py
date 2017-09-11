import unittest
from swutil.collections import *  # @UnusedWildImport

class TestConfig(unittest.TestCase):
    def test(self):
        DefaultDict(lambda i:i)
        OrderedSet(iterable=[1,2])
        RFunction(init_dict={'test':1})
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test1']
    unittest.main()
    