# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:51:55 2019

@author: w
"""

import ffn
from pyetf.data import eod

# retrieve data from eod
etf_tickers = ['SHY','SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
prices = ffn.get(
        tickers=etf_tickers, market='US', 
        provider=eod, 
        start='2013-01-01')

r = prices.to_returns().dropna()
cluster = r.calc_clusters(plot=True)
print(cluster)