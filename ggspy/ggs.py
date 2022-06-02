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
from matrix import log_det

# 3rd-party
import numpy as np

# CardioID


def miu(x):
    return np.mean(x, axis=0)


def sigma(x, lambda_):

    sample_cov = np.cov(x.T, bias=True)
    m, n = sample_cov.shape
    return sample_cov + (lambda_/m)*np.identity(n)


def objective(sigma):

    return -.5*log_det(sigma)


def likelihood(b, x, lambda_):

    like = 0.0
    for b_start, b_end in zip([b[:-1], b[1:]]):
        x_temp = x[b_start:b_end, :]
        temp_sigma = sigma(x_temp, lambda_)
        like += objective(temp_sigma)
    
    return like


def split(x, lambda_):

    orig_like = objective(sigma(x, lambda_))
    max_t = 0
    max_increase = np.NINF

    for t in range(1, len(x)-1):
        sigma_left = sigma(x[:t], lambda_)
        sigma_right = sigma(x[t:], lambda_)
        new_like = objective(sigma_left) + objective(sigma_right)

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

    def step(x, b_old):
        b_new = np.copy(b_old)
        change = False
        for i in range(1, b.size-1):
            t, _ = split(x[b[i-1]:b[b[i+1]]], lambda_)
            t += b[i-1]
            if t != b_old[i]:
                b_new[i] = t
                change = True
        return b_new, change
    
    b_old = b
    change = True

    while not change:
        b_old, change = step(x, b_old)

    return b_old


def gss(x, lambda_, K):

    b = [0, len(x)]
    for k in range(K):
        t, inc = add_point(x, b, lambda_)

        if inc < 0:
            return b[1:-1]
        else:
            i = 0
            for i in range(k):
                if b[i] > t:
                    break
            b = b[:i]+[t]+b[i:]
        
        b = adjust_points(x, b, lambda_)
    
    return b[1:-1]
