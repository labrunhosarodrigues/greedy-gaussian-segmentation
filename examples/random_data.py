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
import plotly.graph_objects as go
import plotly.offline as po


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
    lambda_ = 0.1

    data, breaks = generate(K=K, N=N)

    t11 = timeit.default_timer()
    b, ll1 = ggs.ggs(data, lambda_, K, track=True)
    t12 = timeit.default_timer()

    t21 = timeit.default_timer()
    b2, ll2 = base_ggs.GGS(data=data.T, Kmax=K, lamb=lambda_)
    t22 = timeit.default_timer()

    print(f"{b} -> {t12-t11}\n{b2[-1]} -> {t22-t21}")
    print(f"{ll1}\n{ll2}")

    fig = go.Figure()
    data_traces = data.T
    for trace in data_traces:
        fig.add_trace(
            go.Scatter(
                y=trace,
                mode="lines"
            )
        )
    
    for bb in b[1:-1]:
        fig.add_vline(x=bb, line_color="red", )
    for bb in b2[-1][1:-1]:
        fig.add_vline(x=bb, line_color="blue")
    
    po.plot(fig, filename="randomData_breakpoints.html")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(K+1)),
            y=ll1,
            mode="lines",
            name="ggspy-ggs"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(K+1)),
            y=ll2,
            mode="lines",
            name="cvxggs-ggs"
        )
    )

    po.plot(fig, filename="randomData_LL.html")

