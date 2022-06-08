# -*- coding: utf-8 -*-
"""
ggspy.ggs
-------------

Greedy Gaussian Segmentation.

:copyright: (c) 2022 by CardioID Technologies Lda.
:license: All rights reserved.
"""
# Imports

# built-in

# local

# 3rd-party
import numpy as np
from numpy import linalg

# CardioID


def log_det(matrix):
    """
    Compute the log-determinant of a
    positive definite matrix using its
    Cholensky decomposition.

    Parameters
    ----------
    matrix : n-by-n array
        matrix.

    Returns
    -------
    float
        Log-det of the matrix.
    """
    L = linalg.cholesky(matrix)
    return 2*np.sum(np.log(np.diag(L)))


def miu(x):
    return np.mean(x, axis=0)


miu_vec = np.vectorize(miu)


def sigma(x, lambda_):

    sample_cov = np.cov(x.T, bias=True)
    m, n = x.shape
    return sample_cov + (lambda_/m)*np.identity(n)


sigma_vec = np.vectorize(sigma)


def likelihood(sigma, m):

    return -.5*m*log_det(sigma)


likelihood_vec = np.vectorize(likelihood)


def objective(b, x, lambda_):

    like = 0.0
    segments = np.split(x, b[1:-1])
    sigmas = sigma_vec(segments, lambda_=lambda_)
    like = np.sum(likelihood_vec(sigmas))
    
    return like


def split(x, lambda_):

    m, _ = x.shape
    orig_like = likelihood(sigma(x, lambda_), m)
    max_t = 0
    max_increase = np.NINF
    
    for t in range(1, x.shape[0]-1):
        sigma_left = sigma(x[:t], lambda_)
        sigma_right = sigma(x[t:], lambda_)
        new_like = likelihood(sigma_left, t) + likelihood(sigma_right, m-t)

        if (new_like-orig_like) > max_increase:
            max_increase = new_like-orig_like
            max_t = t
     
    return max_t, max_increase

def add_point(x, b, lambda_):
    candidate_t = np.zeros(b.size-1)
    candidate_inc = np.zeros(b.size-1)
    for i in range(b.size-1):
        t, increase = split(x[b[i]:b[i+1]], lambda_)
        candidate_t[i] = t + b[i]
        candidate_inc[i] = increase
    ind = np.argmax(candidate_inc)

    return candidate_t[ind], candidate_inc[ind]


def adjust_points(x, b, lambda_):

    def step():
        change = False
        for i in range(1, b.size-1):
            t, _ = split(x[b[i-1]:b[i+1]], lambda_)
            t += b[i-1]
            if t != b[i]:
                b[i] = t
                change = True
        return change

    while step():
        continue

    return b


def ggs(x, lambda_, K):

    T, n = x.shape

    b = np.zeros(K+2, dtype=np.int32)
    b[1] = T
    for k in range(2, K+2):
        # find best new breakpoint
        t, inc = add_point(x, b[:k], lambda_)

        # insert new breakpoint if it improves objective
        if inc <= 0:
            return b[1:k]
        else:
            i = np.searchsorted(b[:k], t)
            b[i+1:k+1] = b[i:k]
            b[i] = t
        
        # adjust breakpoints
        b[:k+1] = adjust_points(x, b[:k+1], lambda_)
    
    return b[1:-1]
