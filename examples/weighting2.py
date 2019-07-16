# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:08:56 2019

@author: w
"""
import __init__
import ffn
from pyetf.data import eod
from pyetf.alloc import rpw_standard, rpw_ledoit_wolf
from pyetf.alloc import rpw_garch
from pyetf.alloc import rpw_future
from pyetf.alloc import to_weights
from pyetf.figure import plot_chart

etf_tickers=['SHY','SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
basic_tickers = ['SHY','SPY']
#etf_tickers = basic_tickers 
mc_budget = [0.05, 0.55]

# retrieve data from eod and combine
start_date_str = '2013-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)

# calc portfolio weights
w1 = prices.to_weights(func_weighting=rpw_standard, risk_weights=mc_budget).dropna()
w2 = prices.to_weights(func_weighting=rpw_garch, risk_weights=mc_budget).dropna()
w3 = prices.to_weights(func_weighting=rpw_future, hist_length=-30, risk_weights=mc_budget).dropna()
# calc portfolio performance
pf1 = prices.to_NAV(w1)   
pf2 = prices.to_NAV(w2)
pf3 = prices.to_NAV(w3)
pf1.calc_stats().display()
pf2.calc_stats().display()
pf3.calc_stats().display()

# plot portfolio
pl = pf3.copy()
w = w3.copy()
pl.dropna()
pl = pl.rebase()
w.dropna()
plot_chart(pl[['NAV','spy']], sub_fill=w)   