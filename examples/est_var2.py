# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:01:42 2019
"""
import __init__
import numpy as np
import ffn
from pyetf.data import eod
from pyetf.algos import forecast_var_from_constant_mean
from pyetf.algos import forecast_var_from_garch

# retrieve data from eod
etf_tickers = ['2800']
prices = ffn.get(
        tickers=etf_tickers, market='HK', 
        provider=eod, 
        start='2003-01-01')
hln = 200
ts = 100*prices['2800'].to_returns().dropna()

# constant_mean method
hts = ts.loc['2003'].values
var1 = []
# rolling estimate var
while (len(hts)<len(ts)):
    f_var1, r1 = forecast_var_from_constant_mean(hts[-hln:])
    var1.append(f_var1)
    hts = np.append(hts, ts.iloc[len(hts)])
    
# garch method
hts = ts.loc['2003'].values
var2 = []
# rolling estimate var
while (len(hts)<len(ts)):
    f_var2, r2 =  forecast_var_from_garch(hts[-hln:])
    var2.append(f_var2)
    hts = np.append(hts, ts.iloc[len(hts)])
    