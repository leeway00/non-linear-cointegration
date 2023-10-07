import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
import statsmodels.tsa.stattools as ts

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