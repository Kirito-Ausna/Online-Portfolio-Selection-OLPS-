# Best_OLMAR Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
import cvxpy as cp

span_t = 120


def Best_OLMAR_weight_compute(n, context=None):
    Window = 30  # 与BAH(OLMAR)算法的思想一致
    Win_num = 28
    bt_1 = context['wk']
    if bt_1 is None:
        b1 = np.ones((n, 1)) / n
        E = np.ones((1, Win_num))
        context['history_weight'] = np.dot(b1, E)
        b1 = b1.reshape(n, )
        return b1
    bt_1 = bt_1.reshape((n, 1))
    X_t_1 = context['R'][-1]
    P_t_1 = context['P'][-1]
    P = context['P']
    epsi = 10.0
    history_weight = context['history_weight']
    weight_for_Portfolio = np.dot(X_t_1, history_weight)
    index = np.argpartition(weight_for_Portfolio, -5)[-5:]  # 与BAH(OLMAR)的不同之处，选择出过去一轮表现最好的五个参数，在本轮
    for window_size in range(3, Window + 1):  # 选择这五个参数对应的portfolio值进行算术加权
        Pw = P[-window_size:]
        MAt = (1 / window_size) * np.sum(Pw, axis=0)
        X_t_pred = (MAt / P_t_1).reshape(1, n)
        X_t_P_avarage = float(np.sum(X_t_pred)) / n
        dist = np.linalg.norm(X_t_pred - X_t_P_avarage * np.ones((1, n)))
        difference = epsi - float(np.dot(X_t_pred, bt_1))
        lam = max(0.0, difference / (dist ** 2))
        bt = bt_1 + lam * (X_t_pred.T - X_t_P_avarage * np.ones((n, 1)))
        bt = bt.reshape(n, )
        b = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(b - bt))
        constraints = [cp.sum(b) == 1,
                       b >= 0]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        w = np.array(b.value)
        w = w.reshape(n, )
        context['history_weight'][:, window_size - 3] = w

    current_weight = context['history_weight']
    w = current_weight[:, index]  # 胜者加权，follow the winners
    w = np.sum(w, axis=1)/5
    w = w.reshape(n, )
    return w


if __name__ == "__main__":
    print("this is Best_OLMAR Portfolio")