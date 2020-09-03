# BAH_OLMAR Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
import cvxpy as cp

span_t = 120


def BAH_OLMAR_weight_compute(n, context=None):
    Window = 30  # 设置BAH(OLMAR)的相关参数
    Win_num = 28
    bt_1 = context['wk']
    if bt_1 is None:  # 初始化portfolio
        b1 = np.ones((n, 1)) / n
        E = np.ones((1, Win_num))
        context['history_weight'] = np.dot(b1, E)  # 初始化上一轮的portfolio，以便随后评估历史表现
        b1 = b1.reshape(n, )
        return b1
    bt_1 = bt_1.reshape((n, 1))  # 获取必要的信息
    X_t_1 = context['R'][-1]
    P_t_1 = context['P'][-1]
    P = context['P']
    epsi = 10.0  # 设置每轮获利阈值
    history_weight = context['history_weight']  # 获取上一轮各个参数对应的portfolio
    weight_for_Portfolio = np.dot(X_t_1, history_weight)  # 计算历史表现
    S = float(np.sum(weight_for_Portfolio))
    weight_for_Portfolio = weight_for_Portfolio/S  # 根据历史表现加权，获得权重
    for window_size in range(3, Window + 1):  # 对28个参数对应的OLMAR算法独立运行
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
        context['history_weight'][:, window_size-3] = w  # 保存各个OLMAR算法计算的结果，以便下一轮计算历史表现以及产生最后结果

    current_weight = context['history_weight']
    w = np.dot(current_weight, weight_for_Portfolio.T)  # 对各个参数对应的portfolio加权求和，获得最终结果
    w = w.reshape(n, )
    return w


if __name__ == "__main__":
    print("this is BAH_OLMAR Portfolio")
