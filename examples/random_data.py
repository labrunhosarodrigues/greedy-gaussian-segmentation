# -*- coding: utf-8 -*-
"""
examples.random_data
-------------

GGS example with randomly generated data.

:copyright: (c) 2022 by CardioID Technologies Lda.
:license: All rights reserved.
"""
# Imports

# built-in

# local
from ggspy import ggs

# 3rd-party
import numpy as np

# CardioID


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

    b = ggs.ggs(data, lambda_, K)

    print(f"{breaks}\n{b}")

