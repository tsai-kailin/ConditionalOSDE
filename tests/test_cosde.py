"""test the COSDE"""
# Author: Katherine Tsai <kt14@illinois.edu>
#        
# License: MIT License
import unittest
import numpy as np
from cosde.cosde import MultiCOSDE
from sklearn.gaussian_process.kernels import RBF

class COSDETest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(COSDETest, self).__init__(*args, **kwargs)
        kernelx = RBF(1.)
        kernely = RBF(1.)
        self.f_xy = MultiCOSDE([kernelx], [kernely])
        self.mu1 = 0
        self.mu2 = 5
        self.sigma11 = 1
        self.sigma12 = 0.5
        self.sigma22 = 1
        data = np.random.multivariate_normal(np.array([self.mu1,self.mu2]), 
                                            np.array([[self.sigma11, self.sigma12], [self.sigma12, self.sigma22]]), size=500)
        data_x = data[:,0].reshape((-1,1))
        data_y = data[:,1].reshape((-1,1))

        self.f_xy.fit([data_x], [data_y], 1, max_r=20)
    def test_pdf(self):

        sample_y = np.random.uniform(2,8, size=(1,1))
        sample_x = np.random.normal(self.mu1, self.sigma11, size=(1,1))
        pdf = self.f_xy.get_pdf([sample_x], [sample_y], 1e-3)
        self.assertTrue(pdf >= 0)
    
    def test_cdf(self):
        sample_y = np.array([self.mu2]).reshape(1,1)
        new_x = np.linspace(-10,10,100)
        x0 = -10
        cdf = 0
        for x in new_x:
            cdf += self.f_xy.get_pdf([x.reshape(1,1)], [sample_y], 1e-3) * (x-x0)
            x0 = x
        self.assertTrue(cdf >= .9)
        self.assertTrue(cdf <= 1.1)



if __name__ == 'main':
    unittest.main()