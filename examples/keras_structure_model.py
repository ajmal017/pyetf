# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:13:02 2019

@author: w
"""
import __init__
import ffn
from pyetf.data import eod
from pyetf.algos import strucutre_keras_model
etf_tickers = ['SHY','SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
start_date_str = '2002-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)

for e in prices.columns:
    strucutre_keras_model(prices[e])