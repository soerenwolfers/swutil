'''
Test swutil.v_function
'''
import unittest
from swutil.collections import VFunction

class TestVFunction(unittest.TestCase):

    def test_lambdas(self):
        A = lambda x: x ** 2
        B = lambda x:-x ** 2 + 1
        F = VFunction(A)
        G = VFunction(B)
        H = F + G
        self.assertEqual(H(5.), 1.)

if __name__ == "__main__":
    unittest.main()
