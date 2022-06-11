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
    return 2*linalg.slogdet(L)[1]


def miu(x):
    return np.mean(x, axis=0)


def sigma(x, lambda_):
    sample_cov = np.cov(x.T, bias=True)
    m, n = x.shape
    return sample_cov + (lambda_/m)*np.identity(n)


def likelihood(sigma, m):

    return -.5*m*log_det(sigma)


def objective(b, x, lambda_):
    like = 0.0
    segments =np.split(x, b[1:-1])
    lengths = np.diff(b)
    for i in range(len(segments)):
        like = like + likelihood(sigma(segments[i], lambda_), lengths[i])
    
    return like


def split(x, lambda_):

    m, n = x.shape
    orig_like = likelihood(sigma(x, lambda_), m)
    max_t = 0
    max_increase = np.NINF
    
    left_miu = x[0]
    left_S = np.outer(x[0], x[0])

    right_miu = miu(x[1:])
    right_S = (m-1)*(np.cov(x[1:].T, bias=True) + np.outer(right_miu, right_miu))
    

    for t in range(1, m-1):
        sigma_left = (left_S/t - np.outer(left_miu, left_miu)) + (lambda_/t)*np.identity(n)
        sigma_right = (right_S/(m-t) - np.outer(right_miu, right_miu)) + (lambda_/(m-t))*np.identity(n)
        new_like = likelihood(sigma_left, t) + likelihood(sigma_right, m-t)

        if (new_like-orig_like) > max_increase:
            max_increase = new_like-orig_like
            max_t = t
        
        contribution = np.outer(x[t], x[t])

        left_S = left_S + contribution
        left_miu = left_miu + (x[t]-left_miu)/(t+1)
        
        right_S = right_S - contribution
        right_miu = right_miu - (x[t]-right_miu)/(m-t-1)
        sigma_right = (right_S/(m-t-1) - np.outer(right_miu, right_miu)) + (lambda_/(m-t-1))*np.identity(n)
        
     
    return max_t, max_increase

def add_point(x, b, lambda_):
    candidate_t = -1
    candidate_inc = np.NINF
    position = 0
    
    for i in range(b.size-1):
        t, increase = split(x[b[i]:b[i+1]], lambda_)
        if increase > candidate_inc:
            candidate_t = t + b[i]
            candidate_inc = increase
            position = i+1

    return candidate_t, candidate_inc, position


def adjust_points(x, b, lambda_):
    changed = {i: True for i in range(1, b.size-1)}
    changed[0] = False
    changed[b.size-1] = False
    def step():
        change = False
        for i in range(1, b.size-1):
            if not (changed[i-1] or changed[i+1]):
                continue
            t, _ = split(x[b[i-1]:b[i+1]], lambda_)
            t += b[i-1]
            if t != b[i]:
                b[i] = t
                change = True
                changed[i] = True
            else:
                changed[i] = False
        return change

    while step():
        continue

    return b


def ggs(x, lambda_, K, track=False):

    T, _ = x.shape

    b = np.zeros(K+2, dtype=np.int32)
    b[1] = T
    if track:
        _obj = [objective(b[:2], x, lambda_)]
    for k in range(2, K+2):
        # find best new breakpoint
        t, inc, pos = add_point(x, b[:k], lambda_)

        # insert new breakpoint if it improves objective
        if inc <= 0:
            return b[1:k]
        else:
            b[pos+1:k+1] = b[pos:k]
            b[pos] = t
        
        # adjust breakpoints
        b[:k+1] = adjust_points(x, b[:k+1], lambda_)

        if track:
            _obj.append(objective(b[:k+1], x, lambda_))
    
    if track:
        return b, _obj
    else:
        return b, None
