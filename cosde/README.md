# Implementation Documents


## Function Class `Eigenbase` and `LSEigenBase` in `base.py`

Construct a function class `EigenBase` that returns the following function:
$$
f(x) = \sum_{n=1}^N w_nK(x_n,x),
$$
where $K(\cdot,\cdot)$ is a positive definite kernel function and $w_n$ is the weight for data point $x_n$.

`LSEigenBase` is a function class of linear sum of `EigenBase` objects. Suppose that $f_1,\ldots,f_r$ are `EigenBase` objects, then a `LSEigenBase` object returns $\sum_{i=1}^r a_r f_r(x)$. `LSEigenBase` also takes the multivariate case. Given $f_1(x),\ldots,f_r(x)$ and $g_1(w),\ldots,g_r(w)$, one can construct $\sum_{i=1}^ra_if_i(x)g_i(w)$.


## Function Class `MultiOSDE` in `osde.py`

Construct the multivariete orthonormal series density estimator
For univariate setting, please see [[1]](#1). For multivariate setting, please see Section 7.7.2 in the brainTex.

Given multivariate data $(x^1,\ldots,x^k)$, MultiOSDE outputs the empirical density estimation $\hat{f}(x^1,\ldots,x^k)$.


### step-to-step guide of MultiOSDE.fit()

We want to fit
$$
f(x^1,\ldots, x^k) = \sum_{i=1}^\infty s_i \psi^1_i(x^1)\times\cdots\times\psi^m_i(x^m).
$$
Consider the univariate case and we drop the supscript of $x^1$ for simplicity.
1.   Get the singular function from Gram matrix (set `max_r` to return top-r singular functions). Each singular function is a `EigenBase` object: the $j$-th singular function is represented as
$$
\hat{\psi}_j(x)=\frac{\sqrt{N}}{\lambda_j}\sum_{n=1}^N v_{nj}K(x_n,x).
$$
Since the singular function in the above display is not unit l2-norm, it has to be rescaled.
2.   Normalize the eigenfunction (unit l2-norm). This step depends on the kernel we're using. As of now, only the normalization of Gaussian kernel is implemented. (we'll leave this to future work)

Below is the derivation of the normalization. For simplicity, we drop the index $j$ and let $a_n = (\sqrt{N}v_{nj})/\lambda_j$. Write
$$
\|\hat\psi\|^2 = \langle \hat\psi,\hat\psi\rangle = \sum_{m,n}a_na_m\langle K(x_n,x), K(x_m,x)\rangle.
$$
Note that
\begin{align*}
\int K(x_n,x)K(x_m,x)dx &= \int \exp\left(\frac{-\|x_n-x\|^2}{2l^2}\right)\exp\left(\frac{-\|x_m-x\|^2}{2l^2}\right)dx\\
 & = \sqrt{\pi l^2}\exp\left(\frac{-\|x_n-x_m\|^2}{2(\sqrt{2}l)^2}\right)\\
 & = \sqrt{\pi l^2}\tilde{K}(x_n,x_m),
\end{align*}
where $\tilde{K}$ is the Gaussian kernel with length-scale $\sqrt{2}l$.
Hence, we have
$$
\|\hat\psi\|^2 = \sqrt{\pi l^2}{\bf a}^\top\tilde{K}{\bf a},
$$
with $\tilde{K}$ being the Gram matrix.
3.   Truncate the singular value (set `tol`). $s_i$ decays fast with $i$, we truncate components whose singular value is smaller than `tol`.
4.   Normalize so that the cdf is 1 (unit l1-norm)
Write
\begin{align*}
\|\hat f\|_{L1} &= \int |\sum_{i=1}^r \hat{s}_i\prod_{j=1}^m\hat\psi_i^j(x^j)|dx\\
&= \int \sum_{i=1}^r \hat{s}_i\prod_{j=1}^m\hat\psi_i^j(x^j)dx.
\end{align*}
Recall that
$$
\psi_i^j(x^j) = \sum_{n=1}^N a_{in}^jK(x_n^j,x^j)
$$
and
$$
\int K(x_n^j,x)dx^j = \int \exp\left(\frac{-\|x_n^j-x^j\|^2}{2(l^j)^2}\right)dx=\sqrt{2\pi (l^j)^2}.
$$
Hence, it follows that
$$
\|\hat f\|_{L1} = \sum_{i=1}s_i\prod_{j=1}^m\sqrt{2\pi}l^j\sum_{n=1}^Na_{in}^j
$$



## Function Class MultiCOSDE  in `cosde.py`
This function class implements the conditional orthgonal series density estimator of $f(x^1,\ldots, x^m\mid y^1,\ldots, y^p)$. It has the form
$$
\hat{f}(x^1,\ldots,x^m\mid y^1,\ldots,y^p)\approx \sum_{j=1}^r\hat{c}^y_j\prod_{i=1}^d\hat{\phi}_j^i(x^i),
$$
where
\begin{align*}
\hat c_j^y &= \left(\prod_{i=1}^d\frac{\hat{\lambda}_j^i}{\sqrt{N}}\right)\sum_{n=1}^N \left(\sum_{m=1}^Na_{m,n}\prod_{i=1}^d v_{jm}^i\right) K(y_n,y)= \left(\prod_{i=1}^d\frac{\hat{\lambda}_j^i}{\sqrt{N}}\right) \tilde{a}_j^\top k_y;\\
\hat{\phi}_j^i(x^i) &= \frac{\sqrt{N}}{\hat\lambda}\sum_{m=1}^N v_{jm}^iK(x^i_m,x^i)
\end{align*}
and
$$
A = [a_{mn}] = (K_y+\lambda I)^{-1},\quad k_y = (K(y_1,y)\cdots K(y_N,y))^\top
$$


### Step-to-step Guide of MultiCOSDE.fit()


1.   Get the singular function from Gram matrix
2.   Normalize the singular function to unit l2 norm
3.   Construct the j-th singular value, which is a vector of y: $\left(\prod_{i=1}^d\frac{\hat{\lambda}_j^i}{\sqrt{N}}\right) \tilde{a}_j$

### Step-to-step Guide of MultiCOSDE.get_density_function()


1.   Given fixed $y$, call the `get_singular_values()` to get singular values:
and truncate the components whose singular values are smaller than `tol`
2.   Normalize so that the cdf is 1 (unit l1-norm)


## Functions in `utils.py`:

### Inner Product of two EigenBase objects
\begin{align*}
  \left\langle \sum_{n=1}^{N_1} a_nK(x_n,x), \sum_{l=1}^{N_2}b_lK(x_l,x) \right\rangle   &= 
  \sum_{n=1}^{N_1}\sum_{l=1}^{N_2}a_nb_l\langle K(x_n,x), K(x_l,x)\rangle\\
  &= \sum_{n=1}^{N_1}\sum_{l=1}^{N_2}a_nb_l K(x_n,x_l)\\
  &= \sqrt{\pi l^2}{\bf a^\top \tilde{K} b},
\end{align*}
where ${\tilde{\bf K}}=(\tilde{K}(x_n,x_l))_{m,l}$ is the gram matrix where the length-scale is $\sqrt{2}*l$ ($l$ is the length-scale of the original kernel).


### Inner Product of two LSEigenBase objects

Let $f=\sum_{i=1}^r \alpha_i f_i^1\otimes\cdots\otimes f_i^m$, where $f_i^1,\ldots,f_i^m$ for $i=1,\ldots,r$ be EigenBase functions. Similarly, define $g=\sum_{i=1}^r \beta_i g_i^1\otimes\cdots\otimes g_i^m$, where $g_i^1,\ldots,g_i^m$ for $i=1,\ldots,r$ be EigenBase functions. Then, the inner product of $f$ and $g$ is defined as

\begin{align*}
\langle f, g\rangle = {{\bf\alpha}}^\top ({\bf G}^1\odot\cdots\odot {\bf G}^m){{\bf \beta}},
\end{align*}
where ${\bf G}^k=(\langle f^m_i,g^m_j\rangle)_{i,j}\in\mathbb{R}^{r\times r}$ for $k=1,\ldots,m$.



### Least-squares Method


We want to minimize
$$
\{\hat{p}(Y=i)\}_{i=1}^k=\arg\min\left\|
  f(x) - \sum_{i=1}^k f(x| Y=i)p(Y=i)
\right\|_{L_2}^2
$$
Then compute the transformation to obtain $\hat{p}(Y=i)$:
$$
    \begin{bmatrix}
    \hat{p}(Y=1)\\
    \vdots\\
    \hat{p}(Y=k)
    \end{bmatrix}
    =
    \begin{bmatrix}
    \langle f(x| Y=1),f(x| Y=1)\rangle
    &
    \langle f(x| Y=1),f(x| Y=2)\rangle
    &
    \ldots
    &
    \langle f(x| Y=1),f(x| Y=k)\rangle\\
    \vdots&\ddots&&\vdots\\
    \langle f(x| Y=k),f(x| Y=1)\rangle
    &
    \langle f(x| Y=k),f(x| Y=2)\rangle
    &
    \ldots
    &
    \langle f(x| Y=k), f(x| Y=k)\rangle\\
    \end{bmatrix}^{-1}
    \begin{bmatrix}
    \langle f(x), f(x| Y=1)\rangle\\
    \vdots\\
    \langle f(x),f(x| Y=k)\rangle
    \end{bmatrix}
$$


### Construct the composite operator $\hat{\mathfrak{A}}^\dagger\hat{\mathfrak{B}}$:
\begin{align}
\hat{\mathfrak{A}}^\dagger\hat{\mathfrak{B}} 
&= 
\left\{\sum_{i}(\hat{\mu}_i^1)^{-1} \hat{\phi}_i^1(w)\otimes\hat{\psi}_i^1(x)\right\}
\left\{\sum_{j}\hat{\mu}_{j}^2 \hat{\psi}_{j}^2(x)\otimes \hat{\phi}_j^2(w)\right\}\\
&=\sum_{i,j}\frac{\hat\mu_{j}^2}{\hat\mu_i^1}\langle\hat\psi_i^1,\hat\psi_{j}^2\rangle\hat\phi_i^1(w)\hat\phi_{j}^2(w)
\end{align}




















































------------------
<a id="1">[1]</a> 
 Girolami, M. (2002). Orthogonal series density estimation and the kernel eigenvalue problem. Neural computation, 14(3), 669-688.








