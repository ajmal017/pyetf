# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:43:55 2019

@author: w
"""

import __init__
import ffn
from pyetf.data import eod
from pyetf.algos import strucutre_keras_model
from pyetf.keras_model import train_model, addFeatures, addTarget

etf_tickers = ['IVV', 'TLT', 'IEF', 'GLD', 'DBC']; market='US'
#etf_tickers = ['IVV', 'TLT']; market='US'
start_date_str = '2002-01-01'
prices = ffn.get(tickers=etf_tickers, market=market, 
                 provider=eod, 
                 start=start_date_str)

for i in range(len(prices.columns)):
    for j in range(0, i+1):
        strucutre_keras_model(
                train_model, 
                addFeatures, 
                addTarget, 
                prices[prices.columns[i]],
                prices[prices.columns[j]])