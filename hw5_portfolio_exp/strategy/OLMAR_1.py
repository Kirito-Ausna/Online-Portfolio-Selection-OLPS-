# OLMAR Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk and the portfolio returns µk
import numpy as np
import os
from data_load.stocks import Stocks
from trade.portfolio import Portfolio
import cvxpy as cp

span_t = 120


def OLMAR_1_weight_compute(n, context=None):
    bt_1 = context['wk']
    if bt_1 is None: # 初始化
        b1 = np.ones(n)/n
        return b1
    bt_1 = bt_1.reshape((n, 1))
    X_t_1 = context['R'][-1]  # 获取上一轮的price relative vector和price vector
    P_t_1 = context['P'][-1]
    P = context['P']  # 获得一段时间内的price vector
    window_size = 5  # 设置算法参数
    epsi = 10.0
    Pw = P[-window_size:]  # 取出一个视窗内的价格向量
    MAt = (1/window_size)*np.sum(Pw, axis=0)  # 以过去一个视窗内价格向量的算术平均值作为对本轮价格向量的估计值
    X_t_pred = (MAt/P_t_1).reshape(1, n)  # 计算price relative vector的估计值
    # 计算拉格朗日乘子lambda
    X_t_P_avarage = float(np.sum(X_t_pred))/n
    dist = np.linalg.norm(X_t_pred - X_t_P_avarage*np.ones((1, n)))
    difference = epsi-float(np.dot(X_t_pred, bt_1))
    lam = max(0.0, difference/(dist**2))
    # 计算本轮的portfolio
    bt = bt_1 + lam*(X_t_pred.T - X_t_P_avarage*np.ones((n, 1)))
    bt = bt.reshape(n, )
    # 将计算出来的portfolio投影到限制条件约束的可行域中，得到最终符合条件的portfolio，详见OLMAR论文
    # 本次调用cvxpy库解决凸优化的问题
    b = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(b - bt))
    constraints = [cp.sum(b) == 1,
                   b >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    w = np.array(b.value) # 获得最后的portfoiio解
    w = w.reshape(n, )
    return w


if __name__ == "__main__":
    print("this is OLMAR Portfolio")