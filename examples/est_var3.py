# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 19:56:43 2019
"""

import __init__
import numpy as np
import ffn
from pyetf.data import eod
from pyetf.algos import forecast_var_from_lstm
from pyetf.algos import forecast_var_from_garch
from pyetf.algos import forecast_var_from_best

# retrieve data from eod
etf_tickers = ['SPY']
prices = ffn.get(
        tickers=etf_tickers, market='US', 
        provider=eod, 
        start='2013-01-01')
hln = 200
e = etf_tickers[0].lower()
ts = 100*prices[e].to_returns().dropna()

# hist
hts = ts.loc['2013'].values
var1 = []
# rolling estimate var
while (len(hts)<len(ts)):
    f_var1 = hts[-hln:].var()
    var1.append(f_var1)
    hts = np.append(hts, ts.iloc[len(hts)])
var1 = np.append(np.zeros([len(prices)-len(var1),1]), var1)
    
# garch method
hts = ts.loc['2013'].values
var2 = []
# rolling estimate var
while (len(hts)<len(ts)):
    f_var2, r2 =  forecast_var_from_garch(hts[-hln:])
    var2.append(f_var2)
    hts = np.append(hts, ts.iloc[len(hts)])
var2 = np.append(np.zeros([len(prices)-len(var2),1]), var2)
  
#lstm 
var3 = forecast_var_from_lstm(prices[e])

prices['var1'] = var1
prices['var2'] = var2
prices['var3'] = var3*10

from pyetf.figure import plot_chart
plot_chart(prices[['spy']], sub=prices[['var1','var2','var3']])