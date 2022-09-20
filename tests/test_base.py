#description: test the main
import unittest
import numpy as np
from cosde.base import EigenBase, LSEigenBase
from sklearn.gaussian_process.kernels import RBF


class BaseTest(unittest.TestCase):
    def test_EigenBase(self):
        ### test EigenBase
        kernel = RBF(.425) #this is a magic number (0.425)
        data_x = np.random.normal(0,1, size=(150,1))
        weight_x = np.random.uniform(0,1,150)
        weight_x = weight_x / np.sum(weight_x)
        eb = EigenBase(kernel, data_x, weight_x)

        new_x = np.random.normal(0,1, size=(1,1))


        self.assertTrue(eb.eval(new_x) == np.dot(weight_x, kernel(data_x, new_x).squeeze()))

    def test_LSEigenBase(self):
        #create a bunch of EigenBase objects
        eb_list = []
        for i in range(5):
            kernel = RBF(1)
            data_x = np.random.normal(i*(-1)**i,1, size=(10,1))
            weight_x = np.random.uniform(0,1,10)
            ebi = EigenBase(kernel, data_x, weight_x)
            eb_list.append(ebi)

        weight = np.random.uniform(0,1,5)
        weight = weight/np.sum(weight)

        lseb = LSEigenBase(eb_list, weight)
        new_x = np.random.normal(0,1, size=(1,1))
        out_x = []
        for eb in eb_list:
            out_x.append(eb.eval(new_x))
        out_x = np.array(out_x)
        self.assertTrue(lseb.eval(new_x) == np.dot(weight,out_x))




if __name__ == 'main':
    unittest.main()
