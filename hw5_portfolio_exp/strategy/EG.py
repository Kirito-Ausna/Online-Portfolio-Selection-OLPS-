# EG Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio

span_t = 120


def EG_weight_compute(n, context=None):
    wk = context["wk"]  # Compute w0
    w = np.ones(n)
    w = w / n
    if wk is None:  # initializing
        return w  # return w0
    else:
        X_t = context['R'][-1]
        X_t_modified = np.exp(0.05 * X_t.T / float(np.dot(wk, X_t)))  # Compute the wk+1 using wk
        w = wk * X_t_modified
        Z = np.sum(w)
        w = w / Z  # To make sigma wi = 1
        return w


if __name__ == "__main__":
    print("this is EG Portfolio")
