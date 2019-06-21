# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:12:03 2019

@author: w
"""

import ffn
from Data.get_data import eod

etf_tickers = ['SHY', 'SPY']
prices = ffn.get(tickers=etf_tickers, market='US', provider=eod, start='2017-01-01')

stats = prices.calc_stats()
stats.display()
stats.prices.to_drawdown_series().plot()