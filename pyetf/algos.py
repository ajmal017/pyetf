# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 

#Moving Average  
def MA(ds, n):  
    MA = pd.Series(ds.rolling(n).mean(), name = 'MA_' + str(n))   
    return MA

import statsmodels.formula.api as smf
def liner_regression(x,y):
    model = smf.OLS(y,x)
    results = model.fit()
    b = results.params
    R = results.rsquared
    pvalue = results.pvalues
    t='Y=%0.4fX --- R2=%0.2f%% --- p-value=%0.4f' %(b[0], R*100, pvalue[0])
    return b,t

import statsmodels.tsa.api as smt
def seek_garch_model(TS):
    """
    TS is returns of a price-series
    numpy array or array
    # Seek Best GARCH Model
    res_tup = seek_garch_model(ts)
    order = res_tup[1]
    p_ = order[0]
    o_ = order[1]
    q_ = order[2]
    # Using student T distribution usually provides better fit
    am = arch_model(ts, p=p_, o=o_, q=q_, dist='StudentsT')
    res = am.fit(update_freq=5, disp='off')
    fig = res.plot(annualize='D')
    print(res.summary())
    ts_plot(res.resid, lags=30)
    """
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl

from decorator import decorator
@decorator
def forecast_var(model_est_var, *args, **kwargs):                     
    """
    Use historical data (0 to t) to forecast variance at t+1
    via the model (defined in arch)

    Args:
        * args[0]: returns (numpy array or array): Returns for security.

    Returns:
        forecast variance: float
        residuals: array
    """
    if len(args)<1:
        raise Exception("Not Enough Parameters")
    
    m = model_est_var(*args, **kwargs)
    res = m.fit(update_freq=5, disp='off')
    return res.forecast().variance.values[-1][0], res.resid

@forecast_var
def forecast_var_from_constant_mean(returns):
    """
    returns is historical returns
    """
    from arch.univariate import ConstantMean
    return ConstantMean(returns)

@forecast_var
def forecast_var_from_garch(returns):
    """
    returns is historical returns
    """
    from arch import arch_model
    return arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')

@forecast_var
def forecast_var_from_best(returns):
    """
    returns is historical returns
    """
    from pyetf.algos import seek_garch_model
    from arch import arch_model
    res_tup = seek_garch_model(returns)
    order = res_tup[1]
    p_ = order[0]
    o_ = order[1]
    q_ = order[2]
    return arch_model(returns, p=p_, o=o_, q=q_, dist='StudentsT')