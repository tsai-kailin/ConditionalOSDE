"""Utility Functions"""
# Author: Katherine Tsai <kt14@illinois.edu>
#        
# License: MIT License

import numpy as np
import copy 
from cosde.base import LSEigenBase

def inner_product_base(fobj1, fobj2):
  """
  inner product of two EigenBase objects
  Parameters
  ----------------
  fobj1: EigenBase
  fobj2: EigenBase
  ----------------
  """
  params1 = fobj1.get_params()
  params2 = fobj2.get_params()
  data1 = params1['data']
  data2 = params2['data']
  assert(params1['kernel'] == params2['kernel'])
  kernel = params1['kernel']
  l = kernel.get_params()['length_scale']
  new_kernel = copy.deepcopy(kernel)
  new_kernel.set_params(length_scale=l*np.sqrt(2))
  gram = kernel(data1, data2) * np.sqrt(np.pi)*l
  weight1 = params1['weight']
  weight2 = params2['weight']

  t1 = np.einsum('ij, j->i', gram, weight2)
  t2 = np.dot(weight1, t1)
  return t2


def inner_product(fobj1, fobj2):
  """
  inner product of two LSEigenBase objects
  Parameters
  ----------------
  fobj1: LSEigenBase
  fobj2: LSEigenBase
  ----------------
  """
  r1 = len(fobj1.baselist)
  r2 = len(fobj2.baselist)

  if isinstance(fobj1.baselist[0], list):
    num_modes = len(fobj1.baselist[0])
  else:
    num_modes = 1

  G = []
  for i in range(num_modes):
    G_i = np.zeros((r1, r2))
    for j in range(r1):
      for k in range(r2):
        G_i[j, k] = inner_product_base(fobj1.baselist[j][i], fobj2.baselist[k][i])
    G.append(G_i)
  G_all = np.prod(np.array(G), axis=0)
  t1 = np.einsum('i,ij->j', fobj1.coeff, G_all)
  t2 = np.dot(t1, fobj2.coeff)
  return t2


def least_squares(f_wu, f_wx):
  """
  output a vector of probability distribution
  
  Parameters
  ----------
  f_wu: list of LSEigenBase objetcs, [f(W|U=i)]
  f_wx: a LSEigenBase object, f(W|X)

  Returns
  ---------
  f_ux: ndarray
  """

  k = len(f_wu)

  K = np.zeros((k, k))
  for i in range(k):
    for j in range(k):
      K[i,j] = inner_product(f_wu[i], f_wu[j])
  
  y = np.zeros(k)
  for i in range(k):
    y[i] = inner_product(f_wu[i], f_wx)
  
  invK = np.linalg.solve(K, np.eye(k))

  f_ux = np.einsum('ij,j->i', invK, y)

  return f_ux



def compute_AdaggerB_ij(A, B, i, j, k):
  """
  compute the ij-th component of  A^dagger B

  Parameters
  -------------

  A: MultiOSDE object
  B: MultiOSDE object
  i: i-th component of A
  j: j-th component of B
  k: mode, int

  Return
  ----------------------
  d: the ij-th component of  A^dagger B
  """
  # compute mu_j^2/mu_i^1
  mu_ratio = B.get_singular_value(j) / (A.get_singular_value(i)+1e-15) 

  # compute inner product of \hat\psi_i^1 and \hat\psi_j^2
  assert(A.kernel[k-1] == B.kernel[k-1])

  #this scaling is for Gaussian kernel only
  l = A.kernel[k-1].get_params()['length_scale']
  new_l = l*np.sqrt(2)
  new_ker = copy.deepcopy(A.kernel[k-1])
  new_ker.set_params(length_scale=new_l)
  Gram = np.sqrt(np.pi)*l* new_ker(A.data[k-1], B.data[k-1])
  
  t1 = np.einsum('i,ij->j', A.weight[k-1][i-1], Gram)
  t2 = np.dot(t1, B.weight[k-1][j-1])
  dot_result = t2 
  
  return mu_ratio * dot_result

def compute_AdaggerB(A, B, k):
  """
  compute  A^dagger B

  Parameters
  -------------

  A: MultiOSDE object
  B: MultiOSDE object
  k: mode, int

  Return
  ----------------------
  D: matrix, ndarray 
  x_coor: list of left eigen functions, [EigenBase]
  y_coor: list of right eigen functions, [EigenBase]
  """
  imax = A.r
  jmax = B.r
  
  D = np.zeros((imax, jmax))

  for i in range(1, imax+1):
    for j in range(1, jmax+1):

      D[i-1,j-1] = compute_AdaggerB_ij(A, B, i, j, k)
  x_coor = []
  for i in range(1,imax+1):
    x_coor.append(A.get_eigen_function(k, i))
  
  y_coor = []
  for j in range(1, jmax+1):
    y_coor.append(B.get_eigen_function(k,j))

  return D, x_coor, y_coor


def compute_eigen_system(D, y_coor):
  """
  find the eingevalue and eigenfunction

  Parameters
  -------------
  D: square matrix 
  y_coor: coordinate
  """

  w, vh = np.linalg.eig(D)

  eigen_func = []
  for i in range(vh.shape[1]):
    efunc = LSEigenBase(y_coor, vh[:,i])
    eigen_func.append(efunc)
  return w, eigen_func


#if y_coor is an orthonormal basis
def compute_inv_eigen_system(D, y_coor):
  """
  find the eingevalue and eigenfunction

  Parameters
  -------------
  D: square matrix 
  y_coor: coordinate
  """

  w, vh = np.linalg.eig(D)

  eigen_func = []

  vh = np.linalg.solve(vh,np.eye(vh.shape[0]))

  for i in range(vh.shape[1]):
    efunc = LSEigenBase(y_coor, vh[:,i])
    eigen_func.append(efunc)
  return w, eigen_func
    
  



