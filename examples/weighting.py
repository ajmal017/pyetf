# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:12:03 2019

@author: w
"""

import numpy as np
import ffn
from pyetf.data import eod
from pyetf.data import tushare
from pyetf.alloc import rpw_standard, rpw_ledoit_wolf, to_weights

start_date_str = '2018-01-01'
# retrieve data from eod
etf_tickers = ['2833']
prices = ffn.get(tickers=etf_tickers, market='HK', 
                 provider=eod, 
                 start=start_date_str)

# retrieve data from eod and combine
prices = ffn.get(tickers='SPY', market='US', 
                 provider=eod, 
                 existing = prices,
                 start=start_date_str)

# retrieve data from tushare and combine
prices = ffn.get(tickers='510310', market='SH', asset='FD',
                 provider=tushare, 
                 existing = prices,
                 start=start_date_str)

w = prices.copy()
m = 30

for e in w.columns:
    w[e] = np.nan

# 0:m -> m
# python - [0:m] -> [m-1]  

mc = [0.5, 0.3, 0.2]
for t in range(0, len(prices)-m+1):
    p = prices.iloc[t:t+m]    
    w.iloc[t+m-1] = rpw_standard(p, risk_weights=mc)

w2 = to_weights(prices)

w3 = prices.to_weights(risk_weights=mc)

w4 = prices.to_weights(func_weighting=rpw_ledoit_wolf, risk_weights=mc).dropna()

# performance
#w[t-1] -> s[t] -> nav[t]
pf1 = prices.to_NAV(w2)   
s =  prices.to_shares(w2)
pf2 = prices.to_NAV(w2)
pf3 = prices.to_NAV2(w2)  
pf4 = prices.to_NAV2(w4)

pf1.calc_stats().display()
pf2.calc_stats().display()
pf3.calc_stats().display()
pf4.calc_stats().display() 

# plot 
pl = pf4.copy()
w = w4.copy()
pl.dropna()
pl = pl.rebase()
w.dropna()
from pyetf.figure import plot_chart
plot_chart(pl[['NAV','spy']], sub_fill=w)    