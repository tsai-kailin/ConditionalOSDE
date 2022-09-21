"""basic elements"""
# Author: Katherine Tsai <kt14@illinois.edu>
#        
# License: MIT License

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.gaussian_process.kernels import RBF
import scipy.stats as stats
import math


#function class, see README.md for complete description

class EigenBase:
  """
  kernel  base function
  """
  def __init__(self, kernel, data, weight):
    self.kernel = kernel
    self.data = data
    self.weight = weight

  def eval(self, x):
    """
    evaluate the function at point x

    Parameter
    -----------
    x: scaler

    Return
    -----------
    y: sum_{i=1}^n w_i K(x_i, x)
    """
    Kx = self.kernel(self.data, x)
    Kx = np.squeeze(Kx)
    return np.dot(Kx, self.weight) 

  def get_params(self):
    param_dict = {'kernel': self.kernel,
                  'weight': self.weight,
                  'data': self.data}
    return param_dict

#linear sum of EigenBase
class LSEigenBase:
  def __init__(self, baselist, coeff):
    """
    baselist: list of list of Eigenbase object, length of r and each sublist has length of k (the number of variate)
    coeff: weight, 1-D array, length of r
    """
    assert(len(baselist) == coeff.size)
    

    self.baselist = baselist
    self.coeff = coeff
    
  def get_params(self):
    param_dict = {'base_list': self.baselist,
                  'coeff': self.coeff}
    return param_dict    

  def eval(self,xlist):
    """
    evaluate the density at x_list 
    Parameter
    -----------
    x_list: list of variales

    Return
    -----------
    out: sum_{i=1}^r w_i * Eigenbase.eval(x) 
    

    """
    out = 0
    for i, basef in enumerate(self.baselist):
      y = 1
      #thie is for multivariate setting, i.e. \sum a_i f_i(x)_ig(y)
      if isinstance(xlist, list):
        
        for x, f in zip(xlist, basef):
          
          y *= f.eval(x)
      else:
        y = basef.eval(xlist)
      
      out += self.coeff[i] * y
    return out


if __name__ == "__main__":
    ### test EigenBase
    kernel = RBF(.425) #this is a magic number (0.425)
    data_x = np.random.normal(0,1, size=(15000,1))
    weight_x = np.random.uniform(0,1,15000)
    weight_x = weight_x / np.sum(weight_x)
    eb = EigenBase(kernel, data_x, weight_x)
    new_x = np.linspace(-10,10,200)

    f_x = np.zeros(new_x.shape)
    kde_x = np.zeros(new_x.shape)
    for i, x in enumerate(new_x):
        f_x[i] = eb.eval(x.reshape((1,1)))
        kde_x[i] = np.sum(kernel(data_x, x.reshape((1,1))))/data_x.shape[0]
    plt.plot(new_x, f_x, label='single base function')
    plt.plot(new_x, kde_x, label='kernel density estimator')

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 5*sigma, mu + 5*sigma, 200)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label='true function')



    skse = KernelDensity(kernel='gaussian', bandwidth=.2)
    skse.fit(data_x)
    skse_s = skse.score_samples(new_x[:,np.newaxis])
    plt.plot(new_x, np.exp(skse_s), label='sklearn kernel density estimator')

    plt.legend()
    plt.show()
    ### test LSEigenBase

    #create a bunch of EigenBase objects
    eb_list = []
    for i in range(5):
    #np.random.seed(i)
        kernel = RBF(1)
        data_x = np.random.normal(i*(-1)**i,1, size=(10,1))
        weight_x = np.random.uniform(0,1,10)
        ebi = EigenBase(kernel, data_x, weight_x)
        eb_list.append(ebi)

    weight = np.random.uniform(0,1,5)
    weight = weight/np.sum(weight)

    lseb = LSEigenBase(eb_list, weight)
    for i, x in enumerate(new_x):
        f_x[i] = lseb.eval(x.reshape((1,1)))
    plt.plot(new_x, f_x, label='sum of base functions')
    plt.legend()
    plt.title('function')