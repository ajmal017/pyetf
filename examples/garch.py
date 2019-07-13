# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:44:33 2019

@author: w
"""

import __init__
import ffn
from pyetf.data import eod
from pyetf.figure import ts_plot
from pyetf.algos import seek_garch_model
import statsmodels.tsa.api as smt
from arch import arch_model

# retrieve data from eod
etf_tickers = ['2800']
prices = ffn.get(
        tickers=etf_tickers, market='HK', 
        provider=eod, 
        start='2015-01-01')
ts = 100*prices['2800'].to_returns().dropna().values
tss = 100*prices['2800'].to_returns().dropna().values
# stats
#ts_plot(ts, lags=30)
# MA(3)
#mdl = smt.ARMA(ts, order=(0, 3)).fit(maxlag=30, method='mle', trend='nc')
#print(mdl.summary())
#ts_plot(mdl.resid, lags=30)
'''
# GARCH
res_tup = seek_garch_model(ts)
order = res_tup[1]
p_ = order[0]
o_ = order[1]
q_ = order[2]
# Using student T distribution usually provides better fit
am = arch_model(ts, p=p_, o=o_, q=q_, dist='StudentsT')
res = am.fit(update_freq=5, disp='off')
fig = res.plot(annualize='D')
#print(res.summary())
#ts_plot(res.resid, lags=30)
'''
#GARCH(1,1)
#am = arch_model(ts, vol='Garch', p=1, o=0, q=1, dist='Normal')
#res = am.fit(update_freq=5)
#fig = res.plot(annualize='D')
'''
from arch.univariate import ConstantMean
cm = ConstantMean(ts)
res = cm.fit()
fig = res.plot(annualize='D')
'''

from pyetf.algos import forecast_var_from_constant_mean
f_var1, r1 = forecast_var_from_constant_mean(ts)

from pyetf.algos import forecast_var_from_garch
f_var2, r2 = forecast_var_from_garch(ts)

from pyetf.algos import forecast_var_from_best
f_var3, r3 = forecast_var_from_best(ts)