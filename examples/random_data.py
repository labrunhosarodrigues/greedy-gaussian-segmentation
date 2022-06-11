# -*- coding: utf-8 -*-
"""
examples.random_data
-------------

GGS example with randomly generated data.
"""
# Imports

# built-in
import timeit

# local
from ggspy import ggs

# 3rd-party
import numpy as np
from GGS import ggs as base_ggs


def generate_covariance(N):

    A = np.random.rand(N, N)
    S = np.diag(10*np.random.rand(N))
    return A.T@S@A


def generate_mean(N):
    return np.random.uniform(low=-20, high=20, size=N)


def generate(K=5, N=10):
    lengths = np.random.randint(10, 1000, K+1)
    break_points = np.concatenate(([0], np.cumsum(lengths)))
    
    segments = []
    for l in lengths:
        segments.append(
            np.random.multivariate_normal(
                generate_mean(N),
                generate_covariance(N),
                size=l
            )
        )
    
    data = np.concatenate(segments)

    return data, break_points


if __name__ == "__main__":

    K = 9
    N = 3
    lambda_ = 0.001

    data, breaks = generate(K=K, N=N)

    t11 = timeit.default_timer()
    b, ll1 = ggs.ggs(data, lambda_, K, track=True)
    t12 = timeit.default_timer()

    t21 = timeit.default_timer()
    b2, ll2 = base_ggs.GGS(data=data.T, Kmax=K, lamb=lambda_)
    t22 = timeit.default_timer()

    print(f"{breaks}\n{b} -> {t12-t11}\n{b2[-1]} -> {t22-t21}")
    print(f"{ll1}\n{ll2}")

