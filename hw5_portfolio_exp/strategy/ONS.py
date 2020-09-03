# ONS Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
import cvxopt as cvx

span_t = 120


def ONS_weight_compute(n, context=None):
    R = context['R']
    pt_1 = context['wk']
    if pt_1 is None:  # Initializing
        pt = np.ones(n)
        pt = pt / n  # w0 = (1/n,1/n,……,1/n)
        return pt
    else:
        bt_1 = context['bt-1'].reshape(n, 1)  # use bt-2 and At-2(just part of bt-1 and At-1) to compute the wk
        At_1 = context["At-1"]                # it's a skill that uses the dynamic programming
        rt_1 = R[-1].T
        Delta_t_1 = ((1 / np.dot(pt_1.T, rt_1)) * rt_1).reshape(n, 1)  # compute the gradient of log function
        bt_1 = bt_1 + 1.5*Delta_t_1
        At_1 = At_1 + np.dot(Delta_t_1, Delta_t_1.T)  # compute bt-1 and At-1
        context['bt-1'] = bt_1  # To update the value so can we compute the wk+1 next time
        context['At-1'] = At_1
        At_1_inv = np.linalg.inv(At_1)  # To compute the inverse of matrix
        q = (1 / 8) * np.dot(At_1_inv, bt_1)
        P = cvx.matrix(2*At_1)
        Q = cvx.matrix(np.zeros(n).reshape(n, 1))
        G = cvx.matrix(np.identity(n))
        h = cvx.matrix(q)
        A = np.ones(n).reshape(1, n)
        b = np.dot(A, q) - 1
        A = cvx.matrix(A)
        b = cvx.matrix(b)
        cvx.solvers.options['show_progress'] = False
        w = cvx.solvers.qp(P, Q, G, h, A, b)['x']  # To sovle the Quadratic planning problem
        w = np.array(w).reshape(n, 1)
        w = -w + q
        w = w.reshape(n, )  # To make sure the shape of w
        return w


if __name__ == "__main__":
    print("this is ONS Portfolio")
