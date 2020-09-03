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


def OLMAR_2_weight_compute(n, context=None):
    bt_1 = context['wk']
    if bt_1 is None:  # 初始化
        b1 = np.ones(n) / n
        P = context['P']
        Pw = P[-5:]
        MAt = (1 / 5) * np.sum(Pw, axis=0)
        P_t_1 = P[-1:]
        X_t_pred = (MAt / P_t_1).reshape(1, n)  # 初始化估计值(因为随后便要使用），初始化的方法为利用算术MA估计价格
        context['X_pre'] = X_t_pred
        return b1
    X_pre_t_1 = context['X_pre'].reshape(1, n)  # 获取上一轮对本轮的估计值
    bt_1 = bt_1.reshape((n, 1))  # 获取上一轮的资产配置向量portfolio
    X_t_1 = context['R'][-1].reshape(1, n)  # 获取上一轮的price relative vector
    alpha = 0.3  # 设置参数，衰减系数和获利阈值epsi
    epsi = 10
    X_t_pred = alpha * np.ones((1, n)) + (1 - alpha) * (X_pre_t_1 / X_t_1)  # 利用指数加权平均的思想估计本轮价格相对向量
    context['X_pre'] = X_t_pred  # 保存本轮的估计值，以便下一轮调用
    # 计算朗格朗日乘子lambda
    X_t_P_avarage = float(np.sum(X_t_pred)) / n
    dist = np.linalg.norm(X_t_pred - X_t_P_avarage * np.ones((1, n)))
    difference = epsi - float(np.dot(X_t_pred, bt_1))
    lam = max(0.0, difference / (dist ** 2))
    # 以下过程与OLMAR-1一致
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
    return w


if __name__ == "__main__":
    print("this is OLMAR-2 Portfolio")
