"""Utility Functions"""
# Author: Katherine Tsai <kt14@illinois.edu>
#        
# License: MIT License

import numpy as np
import copy 
from cosde.base import EigenBase, LSEigenBase



def inner_product_base(fobj1, fobj2):
  """
  inner product of two EigenBase objects
  Parameters
  ----------------
  fobj1: EigenBase
  fobj2: EigenBase
  
  Return
  ----------------
  t2: float
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

def l2_norm_base(fobj1):
  """
  return norm of EigenBase object
  Parameter
  ----------
  fobj1: EigenBase

  Return
  ----------
  out: float
  """
  out = np.sqrt(max(inner_product_base(fobj1, fobj1), 1e-16))
  return out

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

  G = []
  if isinstance(fobj1.baselist[0], list):
    num_modes = len(fobj1.baselist[0])
    for i in range(num_modes):
      G_i = np.zeros((r1, r2))
      for j in range(r1):
        for k in range(r2):
          G_i[j, k] = inner_product_base(fobj1.baselist[j][i], fobj2.baselist[k][i])
      G.append(G_i)
  else:
    G_i = np.zeros((r1, r2))
    for j in range(r1):
      for k in range(r2):
        G_i[j, k] = inner_product_base(fobj1.baselist[j], fobj2.baselist[k])
    G.append(G_i)

  G_all = np.prod(np.array(G), axis=0)
  t1 = np.einsum('i,ij->j', fobj1.coeff, G_all)
  t2 = np.dot(t1, fobj2.coeff)
  return t2

def l2_norm(fobj1):
  """
  return norm of LSEigenBase object
  Parameter
  ----------
  fobj1: EigenBase

  Return
  ----------
  out: float
  """
  out = np.sqrt(max(inner_product(fobj1, fobj1), 1e-16))
  return out

def least_squares(f_wu, f_wx, rcond=1e-5):
  """
  output a vector of probability distribution
  
  Parameters
  ----------
  f_wu: list of LSEigenBase objetcs, [f(W|U=i)]
  f_wx: a LSEigenBase object, f(W|X)
  rcond: cut-off ratio of the smallest singular value, float
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
  
  #invK = np.linalg.solve(K, np.eye(k))

  #f_ux = np.einsum('ij,j->i', invK, y)
  f_ux, res,rk,s = np.linalg.lstsq(K,y, rcond=rcond)
  print('results: residuals:{} rank:{} singular values{}'.format(res, rk,s))
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

def Gram_Schmidt_base(f_list):
  """
  compute Gram-schmidt procedure:
  Parameters
  ----------
  f_list: list of EigenBase objects to orthogonalize

  Returns
  ----------
  out_list: list of LSEigenBase objects that are orthonormal
  R: coefficient matrix, ndarray
  """
  out_list = []
  k = len(f_list)
  R = np.zeros((k, k))
  for i, f in enumerate(f_list):
    new_coeff = f.get_params()['weight']
    for j, g in enumerate(out_list):

      r = inner_product_base(f,g)
      new_coeff -= r*g.get_params()['weight']
    params = f.get_params()
    new_f = EigenBase(params['kernel'], params['data'], new_coeff)
    l2 = l2_norm_base(new_f)
    new_coeff /= l2
    new_f.set_weight(new_coeff)
    out_list.append(new_f)
  for i, f in enumerate(f_list):
    for j, g in enumerate(out_list):
      R[j,i] = inner_product_base(f,g)
  return out_list, R


def check_baselist(fobj1, fobj2):
  """
  check whether two baselists are identical

  Paramter
  --------
  fobj1: LSEigenBase object
  fobj2: LSEigenBase object

  Return
  ------
  check: True or False
  """
  if isinstance(fobj1.baselist[0], list):
    for k in range(len(fobj1.baselist[0])):
      check_listk = [i !=j for i,j in zip(fobj1.baselist[k], fobj2.baselist[k])]
      if(sum(check_list) > 0):
        return False
    return True
  else:
    check_list = [i !=j for i,j in zip(fobj1.baselist, fobj2.baselist)]
    check = (sum(check_list) == 0)
    return check

def Gram_Schmidt(f_list):
  """
  compute Gram-schmidt procedure:
  constraint: each LSEigenBase object has same base_list
  Parameters
  ----------
  f_list: list of LSEigenBase objects to orthogonalize

  Returns
  ----------
  out_list: list of LSEigenBase objects that are orthonormal
  R: coefficient matrix, ndarray
  """
  
  out_list = []
  k = len(f_list)
  for i in range(k):
    for j in range(i+1,k):
      check = check_baselist(f_list[i], f_list[j])
      assert(check == True)

  R = np.zeros((k, k))
  for i, f in enumerate(f_list):
    new_coeff = f.get_params()['coeff']
    for j, g in enumerate(out_list):

      r = inner_product(f,g)
      new_coeff -= r*g.get_params()['coeff']
    params = f.get_params()
    new_f = LSEigenBase(params['base_list'], new_coeff)
    l2 = l2_norm(new_f)
    new_coeff /= l2
    new_f.set_coeff(new_coeff)
    out_list.append(new_f)
  for i, f in enumerate(f_list):
    for j, g in enumerate(out_list):
      R[j,i] = inner_product(f,g)
  return out_list, R

  



  



