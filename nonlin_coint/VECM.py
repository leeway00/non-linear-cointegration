import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import statsmodels.api as sm    

def adf_valid_test(y):
    return ts.adfuller(y, regression = 'n')[1], ts.adfuller(y, regression = 'c')[1]

def get_johansen(y, p, mode = 0):
    """
    Get the cointegration vectors at 95% level of significance
    given by the trace statistic test.
    """
    jres = coint_johansen(y, mode, p)
    trace_stat = jres.trace_stat                       # trace statistic
    trace_crit = jres.trace_stat_crit_vals             # critical values
    
    r = [1,1]
    if trace_stat[0] > trace_crit[0, 2]:     # 0: 90%  1:95% 2: 99%
        r[0] = 0.01
    elif trace_stat[0] > trace_crit[0, 1]:
        r[0] = 0.05
    elif trace_stat[0] > trace_crit[0, 0]:
        r[0] = 0.1
        
    if trace_stat[1] > trace_crit[1, 2]:
        r[1] = 0.01
    elif trace_stat[1] > trace_crit[1, 1]:
        r[1] = 0.05
    elif trace_stat[1] > trace_crit[1, 0]:
        r[1] = 0.1
        
    jres.r = r
    jres.evecr = jres.evec[:, 0]
    return jres

def get_johansen_result(pair, price_in, price_out, ret_in, ret_out):
    train_return = ret_in[list(pair)]
    ord_return = max(select_order(train_return, 10).selected_orders.values())
    jres_return = get_johansen(train_return, ord_return, mode = 0)

    train_price = price_in[list(pair)]
    ord_price = max(select_order(train_price, 10).selected_orders.values())
    jres_price = get_johansen(train_price, ord_price, mode = 0)
    ratio = jres_price.evec[:,0]
    spread_in = price_in[list(pair)] @ ratio
    spread_out = price_out[list(pair)] @ ratio
    
    in_adf = adf_valid_test(spread_in)
    out_adf = adf_valid_test(spread_out)
    
    result = {'pair': pair, 'stat1_ret': jres_return.r[0], 'stat2_ret': jres_return.r[1],
                'stat1_price': jres_price.r[0], 'stat2_price': jres_price.r[1],
                'in_adf_n': in_adf[0], 'in_adf_c': in_adf[1], 'out_adf_n': out_adf[0], 'out_adf_c': out_adf[1], 'order': ord_price}
    return result



def common_trend_ols(price_data, oos_data, adf = True):
    price_data, oos_data = price_data.dropna(), oos_data.dropna()
    lin_mod = sm.OLS(price_data.iloc[:,0], sm.add_constant(price_data.iloc[:,1])).fit()
    stationary_common = [1, -lin_mod.params[1]]
    print(stationary_common)

    if adf:
        adf_test = adf_valid_test(price_data @ stationary_common)
        print(adf_test)
    m = (price_data @ stationary_common).mean()
    spread = pd.concat([price_data, oos_data], axis=0) @ stationary_common
    spread.plot()
    plt.hlines(m, spread.index[0], spread.index[-1], colors='r', linestyles='dashed')
    plt.vlines(oos_data.index[0], spread.max(), spread.min(), colors='g', linestyles='dashed')
    plt.show()
    
    # lin_mod = sm.OLS(oos_data.iloc[:,0], sm.add_constant(oos_data.iloc[:,1])).fit()
    # stationary_common2 = [1, -lin_mod.params[1]]
    # (oos_data @ stationary_common2).plot()
    return spread 

def common_trend_vecm(price_data, oos_data, adf = True):
    ord = max(select_order(price_data, 6).selected_orders.values())
    stationary_common = coint_johansen(price_data, 0, ord).evec[:,0]
    # stationary_common = VECM(price_data, k_ar_diff=ord, ).fit().beta[:,0]
    print(stationary_common)

    if adf:
        adf_test = adf_valid_test(price_data @ stationary_common, regression = 'n')
        print(adf_test)
    m = (price_data @ stationary_common).mean()
    spread = pd.concat([price_data, oos_data], axis=0) @ stationary_common
    spread.plot()
    plt.hlines(m, spread.index[0], spread.index[-1], colors='r', linestyles='dashed')
    plt.vlines(oos_data.index[0], spread.max(), spread.min(), colors='g', linestyles='dashed')
    plt.show()
    
    # m = (price_data @ stationary_common).mean()
    # k = pd.concat([price_data, oos_data])
    # t = k @ stationary_common
    # t.plot()
    # plt.hlines(m, t.index[0], t.index[-1], colors='r', linestyles='dashed')
    # plt.show()
    
    return spread