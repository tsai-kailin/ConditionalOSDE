"""test the utils"""
# Author: Katherine Tsai <kt14@illinois.edu>
#        
# License: MIT License
import unittest
import numpy as np
from cosde.base import EigenBase, LSEigenBase
from cosde.utils import *
from sklearn.gaussian_process.kernels import RBF


class utilsTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(utilsTest, self).__init__(*args, **kwargs)
        self.kernel=RBF(0.425)
        self.x = np.random.normal(0,1,size=(5,1))
        w1 = np.random.uniform(0,1,5)
        w2 = np.random.uniform(0,1,5)

        self.EBobj1 = EigenBase(self.kernel, self.x, w1)
        self.EBobj2 = EigenBase(self.kernel, self.x, w2) 
        self.EBobj3 = EigenBase(self.kernel, self.x, w1+w2) 

        coeff1 = np.random.uniform(0,1,2)
        coeff1 /= np.sum(coeff1)
        coeff2 = np.random.uniform(0,1,2)
        coeff2 /= np.sum(coeff2)

        self.LSEBobj1 = LSEigenBase([self.EBobj1, self.EBobj2], coeff1)
        self.LSEBobj2 = LSEigenBase([self.EBobj1, self.EBobj2], coeff2)
        self.LSEBobj3 = LSEigenBase([self.EBobj1, self.EBobj2], coeff1+coeff2)

    def test_inner_product_base(self):
        #verify 3 properties of inner product (EigenBase object)

        #<a,b> = <b,a>
        y0 = inner_product_base(self.EBobj1, self.EBobj2)
        y1 = inner_product_base(self.EBobj1, self.EBobj2)
        self.assertTrue(np.abs(y0 - y1) <= 1e-5)
        
        #<a,a> >= 0
        y2 = inner_product_base(self.EBobj1, self.EBobj1)
        self.assertTrue(y2>=0)

        #<2*a,a+b> = 2<a,a> + 2<a,b>
        weight = self.EBobj1.get_params()['weight']
        
        Ebobj1_2 = EigenBase(self.kernel, self.x, weight*2)
        y3 = inner_product_base(Ebobj1_2, self.EBobj3)
        y4 = 2*y2 + 2*y0
        self.assertTrue(np.abs(y3-y4) <= 1e-5)
        

    def test_inner_product(self):
        #verify 3 properties of inner product (LSEigenBase object)
        #construct LSEigenBase Object

        #<a,b> = <b,a>
        y0 = inner_product(self.LSEBobj1, self.LSEBobj2)
        y1 = inner_product(self.LSEBobj2, self.LSEBobj1)
        self.assertTrue(np.abs(y0 - y1) <= 1e-5)
        
        #<a,a> >= 0
        y2 = inner_product(self.LSEBobj1, self.LSEBobj1)
        self.assertTrue(y2>=0)

        #<2*a,a+b> = 2<a,a> + 2<a,b>
        coeff1_2 = self.LSEBobj1.get_params()['coeff']*2
        
        LSEbobj1_2 = LSEigenBase([self.EBobj1, self.EBobj2], coeff1_2)

        y3 = inner_product(LSEbobj1_2, self.LSEBobj3)
        y4 = 2*y2 + 2*y0
        self.assertTrue(np.abs(y3-y4) <= 1e-5)

    def test_l2_norm_base(self):
        y0 = l2_norm_base(self.EBobj1)
        self.assertTrue(y0>=0)
        
    def test_l2_norm(self):

        y0 = l2_norm(self.LSEBobj1)
        self.assertTrue(y0>=0)      

    def test_least_squares(self):
        res = least_squares([self.LSEBobj1], self.LSEBobj1)
        self.assertTrue(np.abs(res- 1.) <= 1e-5)

    def test_gram_schmidt_base(self):
        f_list = [self.EBobj1, self.EBobj2]
        out_list, R = Gram_Schmidt_base(f_list)
        
        #test the EigenBase objects in out_list are pairwise orthonormal
        y0 = inner_product_base(out_list[0], out_list[0])
        self.assertTrue(np.abs(y0-1)  <= 1e-5)
        
        y1 = inner_product_base(out_list[0], out_list[1])
       
        self.assertTrue(np.abs(y1) <= 1e-5)
        y2 = inner_product_base(out_list[1], out_list[1])
        self.assertTrue(np.abs(y2-1) <= 1e-5)

        # test the reconstruction
        y3 = out_list[0].get_params()['weight']*R[0,0] + out_list[1].get_params()['weight']*R[1,0] 
        y3 = np.linalg.norm(self.EBobj1.get_params()['weight']-y3)
        self.assertTrue(y3 <= 1e-5)

        y4 = out_list[0].get_params()['weight']*R[0,1] + out_list[1].get_params()['weight']*R[1,1] 
        y4 = np.linalg.norm(self.EBobj2.get_params()['weight']-y4)
        self.assertTrue(y4 <= 1e-5)

    def test_gram_schmidt(self):
        f_list = [self.LSEBobj1, self.LSEBobj2]
        out_list, R = Gram_Schmidt(f_list)
        
        #test the EigenBase objects in out_list are pairwise orthonormal
        y0 = inner_product(out_list[0], out_list[0])
        self.assertTrue(np.abs(y0-1)  <= 1e-5)
        
        y1 = inner_product(out_list[0], out_list[1])
       
        self.assertTrue(np.abs(y1) <= 1e-5)
        y2 = inner_product(out_list[1], out_list[1])
        self.assertTrue(np.abs(y2-1) <= 1e-5)
  
        # test the reconstruction
        y3 = out_list[0].get_params()['coeff']*R[0,0] + out_list[1].get_params()['coeff']*R[1,0] 
        y3 = np.linalg.norm(self.LSEBobj1.get_params()['coeff']-y3)
        self.assertTrue(y3 <= 1e-5)

        y4 = out_list[0].get_params()['coeff']*R[0,1] + out_list[1].get_params()['coeff']*R[1,1] 
        y4 = np.linalg.norm(self.LSEBobj2.get_params()['coeff']-y4)



if __name__ == 'main':
    unittest.main()