'''
Test swutil.plots
'''
import unittest
import numpy as np
from swutil.plots import plot_convergence
import matplotlib.pyplot as plt

class TestPlots(unittest.TestCase):
    def test_plotConvergence(self):
        order = plot_convergence([1, 2, 4, 8, 16, 32], [2, 0.3, 0.2, 0.135, 0.0525, 0.03125],
                             plot_rate='fit', reference=0)
        self.assertAlmostEqual(order, -0.9, delta=0.1)
    
    def test_ploConvergence2(self):
        X = 2 ** (np.arange(20))
        Y = np.power(X, -0.05)
        Xmod = X[:-1]
        Y_selfconvergence = np.abs(Y[:-1] - Y[-1])
        Y_estimate = np.power(Xmod, -0.7) * 0.85
        Y_truerate = np.power(Xmod, -0.5) * 0.8
        plt.loglog(Xmod, Y_selfconvergence)
        plt.loglog(Xmod, Y_estimate, '--')
        plt.loglog(Xmod, Y_truerate, '--')
        plt.figure(2)
        plot_convergence(X, Y, preasymptotics=False)
        plt.show()
if __name__ == "__main__":
    unittest.main()
