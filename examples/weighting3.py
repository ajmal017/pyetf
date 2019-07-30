# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:30:08 2019

@author: w
"""

import __init__
import ffn
from pyetf.data import eod
from pyetf.alloc import rpw_lstm, rpw_standard, rpw_garch, rpw_future
from pyetf.alloc import to_weights
from pyetf.figure import plot_chart

etf_tickers=['SHY','SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
basic_tickers = ['SHY','SPY']; etf_tickers = basic_tickers; mc_budget = [0.1, 0.9]
#basic_tickers = ['SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']; etf_tickers = basic_tickers; mc_budget = [0.6]
#mc_budget = [0.05, 0.55]

# retrieve data from eod and combine
start_date_str = '2013-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)

minn = 2
maxn = 6

method = 'lstm'

# calc portfolio weights
if method == 'lstm':
    #lstm
    w1 = prices.to_weights(
            func_weighting=rpw_lstm, 
            model="lstm", 
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()
elif method == 'garch':
    #garch
    w1 = prices.to_weights(
            func_weighting=rpw_garch, 
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()
elif method == 'future':
    #future
    w1 = prices.to_weights(
            func_weighting=rpw_future,
            hist_length=-30,
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()

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
