# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:12:03 2019

@author: w
"""

import ffn
from pyetf.data import eod
from pyetf.data import tushare

# retrieve data from eod
etf_tickers = ['2833', '7300']
prices1 = ffn.get(tickers=etf_tickers, market='HK', 
                 provider=eod, 
                 start='2017-01-01')

# retrieve data from eod and combine
prices2 = ffn.get(tickers='SPY', market='US', 
                 provider=eod, 
                 existing = prices1,
                 start='2017-01-01')

# retrieve data from tushare
prices3 = ffn.get(tickers='600000', market='SH', asset='E',
                 provider=tushare, 
                 existing = prices2,
                 mrefresh=True,
                 start='2017-01-01')

# retrieve data from tushare and combine
prices4 = ffn.get(tickers='510310', market='SH', asset='FD',
                 provider=tushare, 
                 existing = prices3,
                 start='2017-01-01')

stats = prices4.calc_stats()
stats.display()
stats.to_csv(path="results.csv")

r = prices4.to_returns().dropna()
w = r.calc_mean_var_weights()
print([f"{x}" for x in w.index])
print([f"{x:.2f}" for x in w])

from pyetf.alloc import mvw_standard, mvw_ledoit_wolf
w2 = mvw_standard(prices4)
print([f"{x:.2f}" for x in w2])

w3 = mvw_ledoit_wolf(prices4)
print([f"{x:.2f}" for x in w3])

pw1 = r.calc_erc_weights(risk_parity_method='ccd')
print([f"{x:.2f}" for x in pw1])

from pyetf.alloc import rpw_standard, rpw_ledoit_wolf
pw2 = rpw_standard(prices4)
print([f"{x:.2f}" for x in pw2])

pw3 = rpw_ledoit_wolf(prices4)
print([f"{x:.2f}" for x in pw3])

pw4 = rpw_ledoit_wolf(prices4, risk_weights=[0.1,0.1,0.1,0.1,0.5])
print([f"{x:.2f}" for x in pw4])

print(r.calc_clusters())