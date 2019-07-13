# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 12:31:35 2019

@author: w
"""

import ffn
from pyetf.data import eod

# retrieve data from eod
etf_tickers = [
        '2800', #HSI ETF
        '0388', #港交所
        '1299', #友邦
        '2388', #中银香港
        '0002', #中电控股
        '0003', #中华燃气
        '0823', #领展
        '0066', #港铁
        '1177', #中国生物制药
        '0700', #腾讯
        '0941'] #中国移动
prices = ffn.get(
        tickers=etf_tickers, market='HK', 
        provider=eod, 
        start='2013-01-01')

r = prices.to_returns().dropna()
cluster = r.calc_clusters(plot=True)
print(cluster)