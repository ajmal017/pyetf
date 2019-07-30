# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:13:02 2019

@author: w
"""
import __init__
import ffn
from pyetf.data import eod
from pyetf.algos import strucutre_keras_model
from pyetf.keras_model import train_model, addFeatures, addTarget

etf_tickers = ['SHY','SPY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
#etf_tickers = ['SPY']
#etf_tickers = ['SHY','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY']
etf_tickers = ['IVV', 'TLT', 'IEF', 'GLD', 'DBC']; market='US'
etf_tickers = ['2800']; market='HK'

start_date_str = '2002-01-01'
prices = ffn.get(tickers=etf_tickers, market=market, 
                 provider=eod, 
                 start=start_date_str)

for e in prices.columns:
    strucutre_keras_model(train_model, addFeatures, addTarget, prices[e])