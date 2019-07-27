# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 08:54:05 2019

@author: w
"""

import __init__
import numpy as np
import ffn
from pyetf.data import eod
from pyetf.algos import forecast_var_from_lstm
from pyetf.figure import plot_chart

bench = 0.05
etf_tickers = ['SHY', 'SPY']
start_date_str = '2013-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)

var = prices.copy()
w = prices.copy()
for e in prices.columns:
    var[e] = forecast_var_from_lstm(prices[e])
    w[e] = 0.0
for i in range(1,len(prices.columns)):    
    for t in range(len(prices)):
        if var[prices.columns[i]].iloc[t]<bench:
            w[prices.columns[i]].iloc[t] = 1.0
        else:
            w[prices.columns[0]].iloc[t] = 1.0

pf = prices.to_NAV(w) 
var['bench'] = bench

# plot portfolio
pl = pf.copy()
pl.dropna()
pl = pl.rebase()
w.dropna()
plot_chart(pl[['NAV', 'spy']], sub=var, sub_fill=w)
pf.calc_stats().display()