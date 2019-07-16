# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:30:08 2019

@author: w
"""

import __init__
import ffn
from pyetf.data import eod
from pyetf.alloc import rpw_lstm
from pyetf.alloc import to_weights
from pyetf.figure import plot_chart

etf_tickers=['SHY','SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
basic_tickers = ['SHY','SPY']
#etf_tickers = basic_tickers; mc_budget = [0.1, 0.9]
mc_budget = [0.05, 0.55]

# retrieve data from eod and combine
start_date_str = '2013-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)

# calc portfolio weights
w1 = prices.to_weights(func_weighting=rpw_lstm, model="lstm", risk_weights=mc_budget).dropna()
# calc portfolio performance
pf1 = prices.to_NAV(w1)   
pf1.calc_stats().display()

# plot portfolio
pl = pf1.copy()
w = w1.copy()
pl.dropna()
pl = pl.rebase()
w.dropna()
plot_chart(pl[['NAV','spy']], sub_fill=w)   