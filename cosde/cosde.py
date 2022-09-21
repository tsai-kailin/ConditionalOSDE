"""Conditional Orthogonal Series Density Estimator"""
# Author: Katherine Tsai <kt14@illinois.edu>
#        
# License: MIT License

import numpy as np
import copy
from cosde.base import LSEigenBase, EigenBase

class MultiCOSDE:
  def __init__(self, xkernel_list, ykernel_list):
    self.fitted = False
    self.xkernel = xkernel_list
    self.ykernel = ykernel_list
    
    self.xdata = []
    self.ydata = []

    self.eigv = []
    self.eigh = []
    self.density_func = None


  def check_fitted(self):
      try:
        if(self.fitted == False):
          raise ValueError('model not fitted')
      except ValueError as err:
        print(err)

  def fit(self, x_list, y_list, lam=1e-3, max_r = 20):
    """
    set model parameters

    Parameters
    ----------
    x: list of dependent data, [(nsamples, nfeatures)]
    y: list of independent data, [(nsamples, nfeatures)]
    lam: regularity constant, float
    tol: smallest eigenvalue allowed, float
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
    self.max_r = max_r
    temp_eigv = []
    temp_eigh = []

    #get the eigen system of the Gram matrix
    for i, x in enumerate(x_list):
      self.xdata.append(x)
      Gram_x = self.xkernel[i](x,x)

      eigv_x,  eigh_x = np.linalg.eigh(Gram_x)
      eigv_x = eigv_x[::-1][0:max_r]
      eigh_x = eigh_x[:,::-1][:, 0:max_r]

      temp_eigh.append(eigh_x) #singular vector for each component
      temp_eigv.append(eigv_x)

    Gram = np.ones((N,N))
    for i, y in enumerate(y_list):
      self.ydata.append(y)
      Gram_y = self.ykernel[i](y,y)
      
      
      Gram *= Gram_y
    inv_Gram = np.linalg.solve(Gram + lam * np.eye(Gram.shape[0]), np.eye(Gram.shape[0]))


    sigv = np.ones((N, max_r))
    sigh = np.ones((N, max_r))

    for eigv, eigh in zip(temp_eigv, temp_eigh):
      sigv = eigv * sigv / np.sqrt(N)

      sigh = eigh * sigh 
    
    sigh = np.einsum('ij,jk->ik', inv_Gram, sigh)

    temp_sigv = sigh * sigv #singular value for each component



    temp_weight = [[] for i in range(len(x_list))]
    #normalization step

    #the eigen function is unit l2 norm
    for i in range(1,max_r+1):
      for k, x in enumerate(x_list):
        l = self.xkernel[k].get_params()['length_scale']
        new_l = l*np.sqrt(2)
        new_ker = copy.deepcopy(self.xkernel[k])
        new_ker.set_params(length_scale=new_l)
        Gram = np.sqrt(np.pi)*l* new_ker(x,x)


        weight = temp_eigh[k][:,i-1] * np.sqrt(self.xdata[k].shape[0]) / (temp_eigv[k][i-1] + 1e-10)
        t1 = np.dot(np.einsum('i,ij->j',weight, Gram), weight)
        
        weight = weight / np.sqrt(max(t1,1e-10))
        temp_sigv[:,i-1] *= np.sqrt(max(t1,1e-10))
        temp_weight[k].append(weight)

    #flip the eigen function so the non-positive function become non-neagtvie function
    

      
    for id in range(max_r):
      neg_list = [(temp_weight[k][id]<=0+1e-3).all() for k in range(len(x_list))] #0.001 is the small tolerance
      if (sum(neg_list) % 2 == 0) and sum(neg_list)>0:
        print('flip functions: ', id)
        idx = [i for i,d in enumerate(neg_list) if d==True]
        for i in idx:
          temp_weight[i][id] = - temp_weight[i][id]
    self.weight = temp_weight
    self.sigv = temp_sigv


    self.eigh = temp_eigh
    self.eigv = temp_eigv







    self.fitted = True

  def get_singular_function(self, k, i):
    self.check_fitted()
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
    assert(i<= self.max_r)

    #create basis obj
    weight = self.weight[k-1][i-1]

    eigenfunc = EigenBase(self.xkernel[k-1], self.xdata[k-1], weight)

    return eigenfunc

  def get_singular_values(self, y_list, tol=1e-3):
    """
    return ith singular values given y_list

    Parameters
    ----------
    y_list: list of y
    tol: truncation threshold, float
    
    Returns
    -----------
    keep_idx: indices of singular components whose singular values are greater than tol (absolute value)
    sigv_y: list of conditional singular values that are greater than tol (absolute value)
    """
    
    self.check_fitted()
    prod_ky = np.ones(self.ydata[0].shape)  
    for i, y in enumerate(y_list):
      kernel = self.ykernel[i]
 
      ky = kernel(self.ydata[i], y)
 
      prod_ky *= ky

    sigv_y = np.einsum('ji,jk->ik', self.sigv, prod_ky)
    
    sigv_y = np.squeeze(sigv_y)
    
    #truncation
    r = min(np.where(np.abs(sigv_y) >= tol)[0].size, sigv_y.size)
    keep_idx = np.argsort(np.abs(sigv_y))[::-1][0:r]

    

    
    #normalize the singular value so that the cdf is 1
    cdf = 0
    for id in keep_idx:
      prod_weight = 1
      for k in range(len(self.xdata)):
        l = self.xkernel[k].get_params()['length_scale']
        sum_weight = np.sum(self.weight[k][id])* l * np.sqrt(2*np.pi)
        #print(sum_weight) #check that sum_weight should be greater than 1
        prod_weight *= sum_weight
        
      cdf += prod_weight*sigv_y[id]
    

    new_sigv = sigv_y[keep_idx]/cdf

    return keep_idx, new_sigv



  def get_density_function(self, y_list, tol):
    """
    return density function given y_list

    Parameters
    ----------
    y_list: list of y
    tol: truncation threshold, float
    
    Returns
    -----------
    pdf: the density function, LSEigenBase object
    """
    self.check_fitted()
    keep_idx, sigv_y = self.get_singular_values(y_list, tol)

          
    base_list = []
    for i in keep_idx:
      sub_list =[]
      for k in range(1, len(self.xdata)+1):
        sub_list.append(self.get_singular_function(k, i+1))
      base_list.append(sub_list)

    return LSEigenBase(base_list, sigv_y)
    
    
  def get_pdf(self, x_list, y_list, tol):
    """
    return density function given y_list evaluated at x_list

    Parameters
    ----------
    x_list: list of x to be evaluated
    y_list: list of y 
    tol: truncation threshold, float
    
    Returns
    -----------
    pdf: the pdf at x_list conditioned on y_list
    """
    self.check_fitted()
    c_pdf = self.get_density_function(y_list, tol)
    return max(c_pdf.eval(x_list), 0)



