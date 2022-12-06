# -*- coding: utf-8 -*-
"""
examples.random_data
-------------

GGS example with randomly generated data.
"""
# Imports

# built-in
import timeit
import datetime

# local
from ggspy import ggs

# 3rd-party
import numpy as np
from GGS import ggs as base_ggs
import plotly.graph_objects as go
import plotly.offline as po

def date_parser(d_bytes):
    s = d_bytes.decode('utf-8')
    return np.datetime64(datetime.datetime.strptime(s, '"%Y-%m-%d"'))

def get_data():
    # filename = "GGS/Data/Returns.txt"
    filename="https://raw.githubusercontent.com/cvxgrp/GGS/master/Data/ReturnsandDates.txt"
    data = np.genfromtxt(filename,delimiter=' ')
    # Select DM stocks, Oil, and GVT bonds
    feats = [1,4,8]
    data = data[:,feats]
    timestamps = np.genfromtxt(filename,delimiter=' ', usecols=(0), dtype=None, converters={0:date_parser})

    return data, timestamps

if __name__ == "__main__":
    K = 10
    lambda_ = 1e-4

    data, timestamps = get_data()

    t11 = timeit.default_timer()
    b, ll1 = ggs.ggs(data.T, K, lambda_, track=True)
    t12 = timeit.default_timer()

    t21 = timeit.default_timer()
    b2, ll2 = base_ggs.GGS(data=data.T, Kmax=K, lamb=lambda_)
    t22 = timeit.default_timer()

    print(f"{b[-1]} -> {t12-t11}\n{b2[-1]} -> {t22-t21}")
    print(f"{ll1}\n{ll2}")

    Labels = ["Stocks", "Oil", "GVT bonds"]
    colors = ["#00CC96", "#AB63FA", "#FFA15A"]
    fig = go.Figure()
    data_traces = data.T
    for trace in data_traces:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=np.cumsum(trace),
                mode="lines",
                name=Labels.pop(),
                line_color = colors.pop()
            )
        )
    
    for bb in b[-1][1:-1]:
        fig.add_vline(x=timestamps[bb], yref="paper", y1=1, y0=0.5, line_color="blue", )
    for bb in b2[-1][1:-1]:
        fig.add_vline(x=timestamps[bb], yref="paper", y1=0.5, y0=0, line_color="red")
    
    po.plot(fig, filename="finantialData_breakpoints.html")

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

    po.plot(fig, filename="finatialData_LL.html")

    # base_cross_val = base_ggs.GGSCrossVal(data.T, Kmax=30, lambList=[1e-6, 1e-4, 1e-2])
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
    # po.plot(fig, filename="finance_base_CV.html")

    # new_cross_val = base_ggs.GGSCrossVal(data.T, Kmax=30, lambList=[1e-6, 1e-4, 1e-2], func=ggs.ggs)
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
    # po.plot(fig, filename="finance_new_CV.html")
    

