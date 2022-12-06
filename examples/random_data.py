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
    lengths = np.random.randint(10, 100, K+1)
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
    N = 20
    lambda_ = 1e-4

    data, breaks = generate(K=K, N=N)

    t11 = timeit.default_timer()
    b, ll1 = ggs.ggs(data.T, K, lambda_, track=True)
    t12 = timeit.default_timer()

    t21 = timeit.default_timer()
    b2, ll2 = base_ggs.GGS(data=data.T, Kmax=K, lamb=lambda_)
    t22 = timeit.default_timer()

    lambdas = [1e-6, 1e-4, 1e-2, 1, 10, 100]

    bs1 = []
    lls1 = []
    bs2 = []
    lls2 = []
    for l in lambdas:
        b, ll1 = ggs.ggs(data.T, K, lambda_, track=True)
        b2, ll2 = base_ggs.GGS(data=data.T, Kmax=K, lamb=lambda_)
        bs1.append(b[-1])
        lls1.append(ll1[-1])
        bs2.append(b2[-1])
        lls2.append(ll2[-1])
    
    i1 = np.argmax(lls1)
    i2 = np.argmax(lls2)

    print(f"{breaks}\n{lambdas[i1]} -> {bs1[i1]}\n{lambdas[i1]} ->{bs2[i2]}")

    fig = go.Figure()
    data_traces = data.T
    for trace in data_traces:
        fig.add_trace(
            go.Scatter(
                y=trace,
                mode="lines",
                #line_color=colors.pop()
            )
        )
    
    for bb in b[-1][1:-1]:
        fig.add_vline(x=bb, line_color="blue", yref="paper", y1=1, y0=0.5)
    for bb in b2[-1][1:-1]:
        fig.add_vline(x=bb, line_color="red", yref="paper", y1=0.5, y0=0)

    po.plot(fig, filename="randomData_lotsOfbreakpoints.html")

    po.plot(
        go.Figure(
            data=[
                go.Scatter(
                    x=lambdas,
                    y=lls1,
                    mode="lines",
                    name="ggspy-ggs"
                ),
                go.Scatter(
                    x=lambdas,
                    y=lls2,
                    mode="lines",
                    name="cvxggs-ggs"
                )
            ],
            layout={"xaxis_title": "lambdas", "xaxis_type":"log"}
        ),
        filename="lambdas_random.html"
    )

    # #colors = ["#00CC96", "#AB63FA", "#FFA15A"]
    # fig = go.Figure()
    # data_traces = data.T
    # for trace in data_traces:
    #     fig.add_trace(
    #         go.Scatter(
    #             y=trace,
    #             mode="lines",
    #             #line_color=colors.pop()
    #         )
    #     )
    
    # for bb in b[-1][1:-1]:
    #     fig.add_vline(x=bb, line_color="blue", yref="paper", y1=1, y0=0.5)
    # for bb in b2[-1][1:-1]:
    #     fig.add_vline(x=bb, line_color="red", yref="paper", y1=0.5, y0=0)
    
    # po.plot(fig, filename="randomData_breakpoints.html")

    # fig = go.Figure()

    # fig.add_trace(
    #     go.Scatter(
    #         x=list(range(len(ll1))),
    #         y=ll1,
    #         mode="lines",
    #         name="ggspy-ggs"
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=list(range(len(ll2))),
    #         y=ll2,
    #         mode="lines",
    #         name="cvxggs-ggs"
    #     )
    # )

    # po.plot(fig, filename="randomData_LL.html")

    # base_cross_val = base_ggs.GGSCrossVal(data.T, Kmax=10, lambList=[1e-6, 1e-4, 1e-2, 1, 10])
    # print(base_cross_val)

    # fig = go.Figure()
    # for lamb, (train, test) in base_cross_val:
    #     fig.add_trace(
    #         go.Scatter(
    #             y=train,
    #             mode="lines",
    #             name=f"\lambda={lamb}, train"
    #         )
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             y=test,
    #             mode="lines",
    #             name=f"\lambda={lamb}, test"
    #         )
    #     )
    # po.plot(fig, filename="random_base_CV.html")

    # new_cross_val = base_ggs.GGSCrossVal(data.T, Kmax=10,  lambList=[1e-6, 1e-4, 1e-2, 1, 10], func=ggs.ggs)
    # print(new_cross_val)

    # fig = go.Figure()
    # for lamb, (train, test) in new_cross_val:
    #     fig.add_trace(
    #         go.Scatter(
    #             y=train,
    #             mode="lines",
    #             name=f"\lambda={lamb}, train"
    #         )
    #     )
    #     fig.add_trace(
    #         go.Scatter(
    #             y=test,
    #             mode="lines",
    #             name=f"\lambda={lamb}, test"
    #         )
    #     )
    # po.plot(fig, filename="random_new_CV.html")
    