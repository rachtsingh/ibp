# A skeleton file if you find it helpful 

# Imports 
import numpy as np
import numpy.random as npr
from numpy.linalg import norm
import scipy.special as sps 
from scipy.misc import logsumexp, factorial
from scipy.stats import matrix_normal, poisson
from numpy.random import multivariate_normal
from copy import copy
from utils import memoized
import math
import pdb
import matplotlib.pyplot as plt
import sys

from memory_profiler import profile

# my code
from simulate import sample_Z_restaurant, weak_limit_approximation, stick_breaking_construction, indian_buffet_process

# --- Helper Functions --- # 
# Collapsed Likelihood 
# log P(X | Z, sigma_A, sigma_x)
def cllikelihood(data_set, Z, sigma_a=5, sigma_n=.1):
    N, K = Z.shape
    D = data_set.shape[1]
    M = np.linalg.inv(np.dot(Z.T, Z) + (sigma_n/sigma_a)**2*np.eye(K))
    sign, logdet = np.linalg.slogdet(M)
    mterm = logdet * (D/2.0)
    traceterm = (-0.5 / sigma_n**2) * np.trace(data_set.T.dot(np.eye(N) - Z.dot(M).dot(Z.T)).dot(data_set))
    sigmaterms = D * (K * np.log(sigma_a) + (N - K) * np.log(sigma_n)) + (N * D / 2.0) * np.log(2 * np.pi)
    return mterm + traceterm - sigmaterms

# Uncollapsed Likelihood 
def ullikelihood(data_set, Z, A, sigma_n=.1):
    N, K = Z.shape
    D = data_set.shape[1]
    varying = -0.5 * (np.trace((data_set - Z.dot(A)).T.dot(data_set - Z.dot(A)))) / (sigma_n**2)
    constant = (N * D)/2.0 * np.log(2 * np.pi * sigma_n**2)
    return varying - constant

# Compute how many times each history occurs [Kh]
def history_count_set(Z):
    pass

def left_order_form(m):
    twos = np.ones(m.shape[0]) * 2.0
    twos[0] = 1.0
    powers = np.cumprod(twos)[::-1]
    values = np.dot(powers, m)
    idx = values.argsort()[::-1]
    return np.take(m, idx, axis=1)

@memoized
def logfacn(N):
    return np.sum(np.log(np.arange(1, N + 1)))

# Prior on Z
def lpriorZ(Z, alpha):
    lof = left_order_form(Z)
    N, K = Z.shape
    sums = np.sum(lof, axis=0)
    kplus = len(np.nonzero(sums)[0])
    kterm = 0.0
    seen = 0
    i = 0
    counts = []
    while seen < kplus and i < N:
        numnew = max(np.max(np.multiply(np.arange(1, K + 1), lof[i])) - seen, 0)
        counts.append(numnew)
        if numnew != 0.0:
            seen += numnew
            kterm += logfacn(numnew)
        i += 1
    alphaterm = kplus * np.log(alpha) - alpha * np.sum(np.divide(np.ones(i), np.arange(1, i + 1)))
    mkterm = 0.0
    for k in range(kplus):
        mkterm += (logfacn(N - sums[k]) + logfacn(sums[k] - 1)) - logfacn(N)
    return alphaterm - kterm + mkterm 

# Prior on A
# P(A | sigma_a)
def lpriorA( A , sigma_a=5):
    K, D = A.shape
    return matrix_normal.pdf(A, mean=np.zeros(A.shape), rowcov=sigma_a**2 * np.eye(K), colcov=np.eye(D))

# Mean function for A 
def mean_A( data_set , Z , sigma_a=5 , sigma_n=.1):
    N, K = Z.shape
    D = data_set.shape[1]
    M = np.linalg.inv(np.dot(Z.T, Z) + (sigma_n/sigma_a)**2*np.eye(K))
    return M.dot(Z.T).dot(data_set)

# --- Resample Functions --- # 
# Resample A 
def resample_A(data_set, Z, sigma_a=5, sigma_n=.1): 
    N, D = data_set.shape
    K = Z.shape[1]
    mu = mean_A(data_set, Z, sigma_a, sigma_n)
    cov = sigma_n**2 * np.linalg.inv(np.dot(Z.T, Z) + (sigma_n/sigma_a)**2*np.eye(K))
    As = []
    for i in range(D):
        As.append(multivariate_normal(mu[:, i], cov))
    return np.vstack(As).T

# here, Z is Z new (truncate to find Zold)
def sample_new_A(data_set, Z, A, newk, sigma_a=5, sigma_n=.1):
    N, D = data_set.shape
    K = Z.shape[1]
    mu = (np.ones((newk, newk)) + (sigma_n/sigma_a)**2*np.eye(newk)).dot(Z[:, -newk:].T).dot(data_set - Z[:, :-newk].dot(A))
    cov = sigma_n**2 * np.linalg.inv(np.ones((newk, newk)) + (sigma_n/sigma_a)**2*np.eye(newk))
    As = []
    for i in range(D):
        As.append(multivariate_normal(mu[:, i], cov))
    return np.vstack(As).T

# Sample next stick length in the stick-breaking representation
# def sample_pi_next( pi_min , alpha , data_count):

#     return pi_next 

# Equation 
def remove_row(M, zi):
    update = (M.dot(zi.T).dot(zi))/(zi.dot(M).dot(zi.T) - 1) * M
    return M - update

def add_row(M, zi):
    update = (M.dot(zi.T).dot(zi))/(zi.dot(M).dot(zi.T) + 1) * M
    return M - update

def recalculate_M(Z, sigma_n, sigma_a, M_cache=None):
    N, K = Z.shape
    M = np.linalg.inv(np.dot(Z.T, Z) + np.eye(K) * (sigma_n**2/sigma_a**2))
    if M_cache is not None:
        print("Error: {}".format(norm(M - M_cache)))
    return M

def renormalize_log_probs(ll):
    ll -= np.max(ll)
    likelihoods = np.exp(ll)
    return likelihoods/np.sum(likelihoods)

def new_columns_collapsed(data_set, Z, alpha, data_index, sigma_a=5, sigma_n=.1, truncation=10):
    probs = np.zeros(truncation)
    poisprobs = np.zeros(truncation)
    N, K = Z.shape

    for i in range(truncation):
        probs[i] = cllikelihood(data_set, Z, sigma_a, sigma_n)
        poisprobs[i] = poisson.pmf(i, alpha/float(N))
        Z = np.hstack((Z, np.zeros(N).reshape(N, 1)))
        Z[data_index][-1] = 1

    probs = renormalize_log_probs(probs)

    poisprobs /= np.sum(poisprobs)
    sampleprobs = np.multiply(probs, poisprobs)
    sampleprobs /= np.sum(sampleprobs)

    Z = Z[:, :-truncation]

    return np.random.choice(truncation, p=sampleprobs)

# Z is Z old
def new_columns_uncollapsed(data_set, Z, A, alpha, data_index, sigma_a=5, sigma_n=.1, truncation=10):
    logprobs = np.zeros(truncation)
    poisprobs = np.zeros(truncation)
    N, K = Z.shape
    D = data_set.shape[1]

    old_xza = data_set - Z.dot(A)

    for i in range(truncation):
        if i == 0:
            logprobs[i] = 0.0
        else:
            w = np.ones((i, i)) + np.eye(i) * (sigma_n**2/sigma_a**2)
            sign, ldet = np.linalg.slogdet(w)
            firstterm = i * D * np.log(sigma_n/sigma_a) - ldet
            secondterm = 0.5 * np.trace(old_xza.T.dot(Z[:, -i:]).dot(np.linalg.inv(w)).dot(Z[:, -i:].T).dot(old_xza)) / sigma_n**2
            logprobs[i] = firstterm + secondterm
        poisprobs[i] = poisson.pmf(i, alpha/float(N))
        
        # update Z
        Z = np.hstack((Z, np.zeros(N).reshape(N, 1)))
        Z[data_index][-1] = 1

    probs = renormalize_log_probs(logprobs)
    poisprobs /= np.sum(poisprobs)
    sampleprobs = np.multiply(probs, poisprobs)
    sampleprobs /= np.sum(sampleprobs)

    Z = Z[:, :-truncation]

    return np.random.choice(truncation, p=sampleprobs)

# --- Samplers --- # 
# Slice sampler from Teh, Gorur, and Ghahramani, fully uncollapsed 
def slice_sampler( data_set , alpha , sigma_a=5 , sigma_n=.1 , iter_count=35 , init_Z=None):
    data_count = data_set.shape[0]
    dim_count = data_set.shape[1] 
    ll_set = np.zeros( [ iter_count ])
    lp_set = np.zeros( [ iter_count ]) 
    Z_set = list()
    A_set = list() 
    
    # Initialize the variables 
    
    # MCMC loop 
    for mcmc_iter in range( iter_count ):
        
        # Sampling existing pi
        
        # Sampling slice_var
            
        # Extending the matrix
        
        # Sampling existing Z

        # Sampling existing A 

        # Compute likelihoods and store 
        ll_set[ mcmc_iter ] = ullikelihood( data_set , Z , A , sigma_n ) 
        lp_set[ mcmc_iter ] = lpriorA( A , sigma_a ) + lpriorZ( Z , alpha ) 
        A_set.append( A ); Z_set.append( Z ) 

        # print
        print mcmc_iter , Z.shape[1] , ll_set[ mcmc_iter ] , lp_set[ mcmc_iter ] 

    # return 
    return Z_set , A_set , ll_set , lp_set

def trim_ZA(Z, A):
    sums = np.sum(Z, axis=0)
    to_keep = np.where(sums != 0)[0]
    return Z[:, to_keep], A[to_keep, :]

# The uncollapsed LG model. In a more real setting, one would want to
# additionally sample/optimize the hyper-parameters!  
def ugibbs_sampler(data_set, alpha, sigma_a=5, sigma_n=.1, iter_count=50, init_Z=None):
    data_count = data_set.shape[0]
    dim_count = data_set.shape[1] 
    ll_set = np.zeros([iter_count])
    lp_set = np.zeros([iter_count]) 
    Z_set = list()
    A_set = list() 
    
    N = data_count # in case I forget again
    # Initialize Z randomly (explore how different initializations matter)
    Z, _ = sample_Z_restaurant(data_count, alpha, 35)

    h = []

    # MCMC loop 
    for mcmc_iter in range(iter_count):
        # Sampling existing A
        A = resample_A(data_set, Z, sigma_a, sigma_n)

        # Sampling existing Z
        for data_index in range(data_count):
            sums = np.sum(Z, axis=0) - Z[data_index]
            for col in range(Z.shape[1]):
                probs = np.zeros(2)
                o = np.array([1 - (sums[col] / float(N)), sums[col] / float(N)])
                Z[data_index][col] = 0
                probs[0] = ullikelihood(data_set, Z, A, sigma_n)
                Z[data_index][col] = 1          
                probs[1] = ullikelihood(data_set, Z, A, sigma_n)
                likelihoods = renormalize_log_probs(probs)
                probs = np.multiply(likelihoods, o)
                probs /= np.sum(probs)

                Z[data_index][col] = np.random.choice(2, p=probs)

            # Consider adding new features
            newk = new_columns_uncollapsed(data_set, Z, A, alpha, data_index, sigma_a, sigma_n)
            h.append(newk)
            if newk > 0:
                Z = np.hstack((Z, np.zeros((N, newk))))
                for i in range(newk):
                    Z[data_index][-(i + 1)] = 1
                Anew = sample_new_A(data_set, Z, A, newk, sigma_a, sigma_n)
                A = np.concatenate((A, Anew), axis=0)

        # Remove any unused
        Z, A = trim_ZA(Z, A)
        
        # Compute likelihood and prior 
        ll_set[mcmc_iter] = ullikelihood(data_set, Z, A, sigma_n) 
        lp_set[mcmc_iter] = lpriorA(A, sigma_a) + lpriorZ(Z, alpha) 
        A_set.append(A); Z_set.append(Z) 

        # print
        print mcmc_iter, Z.shape[1], ll_set[mcmc_iter], lp_set[mcmc_iter] 

    # return 
    return Z_set, A_set, ll_set, lp_set 

def trim_Z(Z):
    sums = np.sum(Z, axis=0)
    to_keep = np.where(sums != 0)[0]
    return Z[:, to_keep]
            
# The collapsed LG model from G&G.  In a more real setting, one would
# want to additionally sample/optimize the hyper-parameters!
def cgibbs_sampler(data_set, alpha, sigma_a=5.0, sigma_n=0.1, iter_count=25, init_Z='ibp'):
    data_count = data_set.shape[0] 
    ll_set = np.zeros([iter_count])
    lp_set = np.zeros([iter_count]) 
    Z_set = list()
    A_set = list()

    N = data_count # in case I forget again 
    
    # Initialize Z randomly (explore how different initializations matter)
    if init_Z == 'ibp':
        Z = indian_buffet_process(N, alpha, 35)
    elif init_Z == 'stick':
        Z = stick_breaking_construction(N, alpha, 35)
    else:
        Z = weak_limit_approximation(N, alpha, 35)

    priors = []
    lls = []
    diffs = []

    # MCMC loop 
    for mcmc_iter in range(iter_count):
        Z = trim_Z(Z)

        for data_index in range(data_count):

            sums = np.sum(Z, axis=0) - Z[data_index]
            for col in range(Z.shape[1]):
                probs = np.zeros(2)
                o = np.array([1 - (sums[col] / float(N)), sums[col] / float(N)])
                Z[data_index][col] = 0
                probs[0] = cllikelihood(data_set, Z, sigma_a, sigma_n)
                Z[data_index][col] = 1          
                probs[1] = cllikelihood(data_set, Z, sigma_a, sigma_n)
                likelihoods = renormalize_log_probs(probs)

                # priors.append(np.log(o[0] + 1e-5) - np.log(o[1] + 1e-5))
                # lls.append(np.log(likelihoods[0] + 1e-5) - np.log(likelihoods[1] + 1e-5))
                # diffs.append(np.absolute(lls[-1]) - np.absolute(priors[-1]))

                probs = np.multiply(likelihoods, o)
                probs /= np.sum(probs)

                Z[data_index][col] = np.random.choice(2, p=probs)

            # Consider adding new features
            new_columns = new_columns_collapsed(data_set, Z, alpha, data_index, sigma_a, sigma_n)
            if new_columns > 0:
                Z = np.hstack((Z, np.zeros((N, new_columns))))
                for i in range(new_columns):
                    Z[data_index][-(i + 1)] = 1
                                   
        # Compute likelihood and also the mean value of A, just so we
        # can visualize it later
        ll_set[mcmc_iter] = cllikelihood(data_set, Z, sigma_a, sigma_n)
        lp_set[mcmc_iter] = lpriorZ(Z, alpha)
        A = mean_A(data_set, Z, sigma_a, sigma_n)
        A_set.append(A); Z_set.append(left_order_form(Z))

        print mcmc_iter, Z.shape[1], ll_set[mcmc_iter], lp_set[mcmc_iter], np.mean(np.absolute(data_set - Z.dot(A)))
        # plt.plot(np.arange(len(diffs)), np.array(diffs), color='green')
        # # plt.plot(np.arange(len(lls)), np.array(lls))
        # plt.show()
        
    return Z_set, A_set, ll_set, lp_set