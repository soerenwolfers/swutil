'''
Test swutil.plots
'''
import unittest
from swutil.plots import plot_convergence

class TestPlots(unittest.TestCase):
    def test_plotConvergence(self):
        order = plot_convergence([1, 2, 4, 8, 16, 32], [2, 0.3, 0.2, 0.135, 0.0525, 0.03125],
                             plot_rate='fit', reference=0)
        self.assertAlmostEqual(order, -0.9, delta=0.1)

if __name__ == "__main__":
    unittest.main()
