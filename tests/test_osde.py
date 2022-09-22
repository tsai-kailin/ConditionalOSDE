"""test the OSDE"""
# Author: Katherine Tsai <kt14@illinois.edu>
#        
# License: MIT License
import unittest
import numpy as np
from cosde.osde import MultiOSDE
from sklearn.gaussian_process.kernels import RBF

class OSDETest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OSDETest, self).__init__(*args, **kwargs)
        kernelx = RBF(.425)

        kernel_list = [kernelx]
        self.f_x = MultiOSDE(kernel_list)
        data_x = np.random.normal(0, 1, size=(1000, 1))
        data_list = [data_x]
        self.f_x.fit(data_list, 1e-3)

    def test_pdf(self):
        new_x = np.random.normal(0, 1, size=(1,1))
        pdf = self.f_x.get_pdf([new_x])
        self.assertTrue(pdf >= 0)
    def test_cdf(self):
        new_x = np.linspace(-10,10,200)
        cdf = 0

        x0 = new_x[0]
        for x in new_x:
            cdf += self.f_x.get_pdf([x.reshape((1,1))])*(x-x0)
            x0 = x
        self.assertTrue(cdf >=0.9)
        self.assertTrue(cdf <=1.1)


if __name__ == 'main':
    unittest.main()