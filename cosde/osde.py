"""Orthogonal Series Density Estimator"""
# Author: Katherine Tsai <kt14@illinois.edu>
#        
# License: MIT License

import numpy as np
import copy
from cosde.base import LSEigenBase, EigenBase

class MultiOSDE:
  def __init__(self, kernel_list):
    self.fitted = False
    self.kernel = kernel_list
    
    self.data = []
    self.eigv = []
    self.eigh = []
    self.density_func = None


  def check_fitted(self):
      try:
        if(self.fitted == False):
          raise ValueError('model not fitted')
      except ValueError as err:
        print(err)

  def fit(self, x_list, tol=1e-3, max_r = 20):
    """
    set model parameters

    Parameters
    ----------
    x: list of data, [(nsamples, nfeatures)]
    tol: smallest eigenvalue allowed, int
    max_r: maximum rank
    """
    
    
    #make sure that the list is empty
    self.data = []
    self.eigv = []
    self.eigh = []  
    self.sigv = []  
    self.weight = [[] for i in range(len(x_list))]
    N = x_list[0].shape[0]
    max_r = min(max_r, N)
    r = x_list[0].shape[0]
    temp_eigv = []
    temp_eigh = []
    for i, x in enumerate(x_list):
      self.data.append(x)
      Gram_x = self.kernel[i](x,x)

      eigv_x,  eigh_x = np.linalg.eigh(Gram_x)
      eigv_x = eigv_x[::-1][0:max_r]
      eigh_x = eigh_x[:,::-1][:, 0:max_r]

      temp_eigh.append(eigh_x) #singular vector for each component
      temp_eigv.append(eigv_x)




    sigv = np.ones(max_r)
    sigh = np.ones((N, max_r))
    for eigv, eigh in zip(temp_eigv, temp_eigh):
      sigv = eigv * sigv / np.sqrt(N)
      sigh = eigh * sigh 
    
    temp_sigv = sigv * np.sum(sigh, axis = 0)/N #singular value for each component

    #make sure the singular value is positive
    for i,v in enumerate(temp_sigv):
      if(v < 0):
        temp_eigh[0][:,i] = -temp_eigh[0][:,i]
        temp_sigv[i] = - temp_sigv[i]
    

    temp_weight = [[] for i in range(len(x_list))]
    #normalization step

    #the eigen function is unit l2 norm
    for i in range(1,max_r+1):
      for k, x in enumerate(x_list):
        l = self.kernel[k].get_params()['length_scale']
        new_l = l*np.sqrt(2)
        new_ker = copy.deepcopy(self.kernel[k])
        new_ker.set_params(length_scale=new_l)
        Gram = np.sqrt(np.pi)*l* new_ker(x,x)


        weight = temp_eigh[k][:,i-1] * np.sqrt(self.data[k].shape[0]) / (temp_eigv[k][i-1] + 1e-10)
        t1 = np.dot(np.einsum('i,ij->j',weight, Gram), weight)
        
        weight = weight / np.sqrt(max(t1,1e-10))
        temp_sigv[i-1] *= np.sqrt(max(t1,1e-10))
        temp_weight[k].append(weight)

    #truncation
    r = min(np.where(temp_sigv >= tol)[0].size, max_r)
    self.r =r
    keep_idx = np.argsort(temp_sigv)[::-1][0:r]


    #store eigenvalue and eigen vector of gram matrix
    for i, eigh in enumerate(temp_eigh):
      self.eigh.append(eigh[:,keep_idx])
      self.eigv.append(temp_eigv[i][keep_idx])
    for k in range(len(x_list)):
      for id in keep_idx:
        self.weight[k].append(temp_weight[k][id])


    #normalize the singular value so that the cdf is 1
    cdf = 0
    for id in keep_idx:
      prod_weight = 1
      for k in range(len(x_list)):
        l = self.kernel[k].get_params()['length_scale']
        sum_weight = np.sum(temp_weight[k][id])* l * np.sqrt(2*np.pi)
        #print(sum_weight) #check that sum_weight should be greater than 1
        prod_weight *= sum_weight
        
      cdf += prod_weight*temp_sigv[id]
    


    self.sigv = temp_sigv[keep_idx] / cdf



    self.fitted = True


  def get_eigen_function(self, k, i):
    """
    return ith singular function of mode k

    Parameters
    ----------
    i:  int
    k:  int, mode
    Return
    -----------
    f_i(x[k]): EigenBase object
    """
    
    self.check_fitted()
    assert(i<= self.r)

    #create basis obj
    weight = self.weight[k-1][i-1]

    eigenfunc = EigenBase(self.kernel[k-1], self.data[k-1], weight)

    return eigenfunc

  
  def get_singular_value(self, r):
    """
    return singular values

    Parameters
    -----------------------
    r: int
    """
    self.check_fitted()
    assert(r<= self.r)
    return self.sigv[r-1]


  def get_density_function(self):
    """
    return density function 
    """  
    if self.density_func == None:
      self.check_fitted()
      base_list = []
      for i in range(1, self.r+1):
        sub_list =[]
        for k in range(1, len(self.data)+1):
          sub_list.append(self.get_eigen_function(k, i))
        base_list.append(sub_list)

      self.density_func = LSEigenBase(base_list, self.sigv)
    return self.density_func

        
  
  def get_pdf(self, newx_list):
    """
    return density function evaluated at x_list

    Parameters
    ----------
    x_list: list of x to be evaluated
    
    Returns
    -----------
    pdf: the pdf at x_list 
    """
    self.get_density_function()
    return self.density_func.eval(newx_list)

