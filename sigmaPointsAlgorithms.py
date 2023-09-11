import numpy as np
from numpy.linalg import cholesky
'''
It may not be obvious why this is 'correct', and indeed,
it cannot be proven that this is ideal for all nonlinear problems.
But you can see that we are choosing the sigma points proportional to the square root of the covariance matrix,
and the square root of variance is standard deviation.
So, the sigma points are spread roughly according to  ±1𝜎 times some scaling factor.
There is an  𝑛 term in the denominator, so with more dimensions the points will be spread out and weighed less.


Reasonable Choices for the Parameters
𝛽=2is a good choice for Gaussian problems,
𝜅=3−𝑛 where  𝑛 is the dimension of  𝐱 is a good choice for  𝜅
0≤𝛼≤1  is an appropriate choice for  𝛼 where a larger value for  𝛼 spreads the sigma points further from the mean.
'''
def Merwes_sigmas(alpha , beta , kappa , state_dim , cov_mat):
    mu = 0
    cov_mat = [[1 , 0]
               [1 , 1]]
    cov_mat_root = cholesky(cov_mat)
    n = state_dim
    Gamma = alpha**2 * (n+kappa) - n
    xi_arr = np.zeros([2*n + 1 , n],dtype='float64')


    # define cov, mean weights for all indecies except 1
    weight = 1/(2*(n+Gamma))
    weight_mean_arr = np.full(2*n + 1, weight)
    weight_cov_arr = np.full(2*n + 1, weight)
    # define cov , mean weights for the first index
    weight_mean_arr[0] = Gamma / (n + Gamma)
    weight_cov_arr[0] = Gamma / (n + Gamma) + 1 - alpha ** 2 + beta

    xi0 = mu
    xi_arr[0] = xi0
    xi_arr[1:n] = mu + (n+Gamma)**.5 * cov_mat_root[1:n]
    xi_arr[n+1,2*n] = mu - (n + Gamma)**.5 * cov_mat_root[n+1:2*n]
