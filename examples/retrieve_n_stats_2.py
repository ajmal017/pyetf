# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:01:54 2019

@author: w
"""

import ffn
from pyetf.data import eod
from pyetf.data import tushare
from pyetf.figure import plot_chart

# retrieve data from eod and combine
etf_tickers = ['IVV', 'GLD', 'VNQ', 'AGG', 'TLT', 'IEF', 'IEI', 'OIL', 'USO', 'DBC']
etf_tickers = ['IVV', 'GLD', 'TLT', 'IEF', 'DBC']
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod,
                 start='2000-01-01')

stats = prices.calc_stats()
stats.display()

pl = prices.copy()
pl.dropna()
pl = pl.rebase()
plot_chart(pl)