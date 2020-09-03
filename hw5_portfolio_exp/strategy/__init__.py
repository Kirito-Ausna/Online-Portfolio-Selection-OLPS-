import datetime

from strategy.Best import Best_weight_compute
from strategy.EW import EW_weight_compute
from strategy.MV import MV_weight_compute
from strategy.EG import EG_weight_compute
from strategy.ONS import ONS_weight_compute
from strategy.OLMAR_1 import OLMAR_1_weight_compute
from strategy.OLMAR_2 import OLMAR_2_weight_compute
from strategy.BAH_OLMAR import BAH_OLMAR_weight_compute
from strategy.Best_OLMAR import Best_OLMAR_weight_compute
import numpy as np

'''
context参数说明

Pk：k时刻的价格
Rk：k时刻的price relative vector
P：k - span_t + 1:k一段时间的价格
R：k - span_t + 1:k一段时间的return
frequency: rebanlance的周期
'''
'''
method说明

"EW": equal weighted
"VW": value weighted
"BH": buy and hold
"MV": minumum variance
"MVWC":Minumum-variance Portfolio with the Constraint
"MC":Mean-CVaR
"MCWC":Mean-CVaR Portfolio with the constraint
"TP_VW": Blending VW MV
"TP_EM": Blending EW MV
"OBP": Orthogonal Bandit Portfolio
"PBTS"
"PCTS" : portfolio choices via Thompson sampling
'''


def methods_config():
    """
    :param name: 算法名称
    :param function: 算法所在的函数名称
    :param duplicate: 实验重复次数
    :param k: PBTS特有参数
    :param stock_size: 一共有几只股票
    :param portfolio_size: 每个组合有几只股票，若0则不限制
    :param update_type: 更新类型，不同算法意义不同
    """
    Best = {"name": "Best", "function": "Best", "data_type": "density"}
    EW = {"name": "EW", "function": "EW", "data_type": "density"}
    MV = {"name": "MV", "function": "MV", "data_type": "density"}
    EG = {"name": "EG", "function": "EG", "data_type": "density"}
    ONS = {"name": "ONS", "function": "ONS", "data_type": "density"}
    OLMAR_1 = {"name": "OLMAR_1", "function": "OLMAR_1", "data_type": "density"}
    OLMAR_2 = {"name": "OLMAR_2", "function": "OLMAR_2", "data_type": "density"}
    BAH_OLMAR = {"name": "BAH_OLMAR", "function": "BAH_OLMAR", "data_type": "density"}
    Best_OLMAR = {"name": "Best_OLMAR", "function": "Best_OLMAR", "data_type": "density"}

    methods = [Best, EW, EG, ONS, OLMAR_1, OLMAR_2]
    methods_name = ["Best", "EW", 'EG', 'ONS', 'OLMAR_1', 'OLMAR_2']

    return methods, methods_name


def datasets_config():
    # !!!根据特征工程，init_t一定一定要大于12个单位
    # ff25_csv = {"name": "ff25_csv", "filename": "portfolio25.csv", "span_t": 120, "init_t": 20, "frequency": "month"}
    ff49_csv = {"name": "ff49_csv", "filename": "portfolio49.csv", "span_t": 120, "init_t": 20, "frequency": "month"}
    NYSE = {"name": "NYSE", "filename": "NYSE.txt", "span_t": 120, "init_t": 20, "frequency": "none"}
    ff49_csv = {"name": "ff49_csv", "filename": "portfolio49.csv", "span_t": 120, "init_t": 20, "frequency": "month"}
    ff100_csv = {"name": "ff100_csv", "filename": "portfolio100.csv", "span_t": 120, "init_t": 20, "frequency": "month"}
    datasets = [NYSE, ff49_csv, ff100_csv]
    dataset_name = ["NYSE", "ff49_csv", 'ff100_csv']
    return datasets, dataset_name


def runPortfolio(stocks, portfolio, method, dataset):
    # get stock data
    m = stocks.Nmonths
    n = stocks.Nportfolios
    R = stocks.portfolios
    P = stocks.portfolios_price

    MF = stocks.market_feature

    SF = stocks.stock_feature

    zero = np.zeros(n).reshape(n, 1)
    I_n = np.identity(n)
    weight_compute = eval(method["function"] + "_weight_compute")
    context = {"frequency": portfolio.frequency, "return_list": [], "wk": None, "bt-1": zero, 'At-1': I_n,
               'X_pre': None, 'history_weight': None}

    for k in range(dataset["span_t"] - 1 + dataset["init_t"], m, 1):
        context["Pk"] = P[k]
        context["Rk"] = R[k]
        context["MF"] = MF[k]
        context["SF"] = SF[k * n:(k + 1) * n, :]
        context["next_Rk"] = None
        if k < m - 1:
            context["next_Rk"] = R[k + 1]
        context["P"] = P[k - dataset["span_t"] + 1: k]
        context["R"] = R[k - dataset["span_t"] + 1: k]
        wk = weight_compute(n, context)
        context['wk'] = wk
        portfolio.rebalance(target_weights=wk)

        context["return_list"].append(portfolio.return_list[-1])


if __name__ == "__main__":
    print("this is config and run script, start please go to main.py")
