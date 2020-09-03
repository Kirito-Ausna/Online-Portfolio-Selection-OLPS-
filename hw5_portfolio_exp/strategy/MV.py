# MV Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
from cvxopt import solvers, matrix

span_t = 120


def MV_weight_compute(n, context=None):
    R = context['R'] - 1  # get the price relative vector of a period
    Rt = R[-1]  # the price relative vector of time t
    E = np.identity(n)
    G = matrix(-1 * E)
    h = matrix(np.zeros(n))
    Q = np.cov(R, rowvar=False)  # To compute the covarience matrix
    P = matrix(2 * Q)
    q = matrix(-0.05 * Rt)
    A = matrix(np.ones((1, n)))
    b = matrix([1.0])
    solvers.options['show_progress'] = False
    w = solvers.qp(P, q, G, h, A, b)  # use MVP model to compute wk
    w = np.array(w['x']).reshape(36, )  # Make sure the shape of wk
    return w


if __name__ == "__main__":
    print("this is MV Portfolio")
