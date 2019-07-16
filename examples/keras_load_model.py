# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:55:27 2019

@author: w
"""

import __init__
import numpy as np
import ffn
from pyetf.data import eod
from pyetf.algos import load_keras_model

etf_tickers = ['SPY']
start_date_str = '2013-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)

for e in prices.columns:
    # Load Model
    x_dataset, y_dataset, model = load_keras_model(prices[e])
    # Forecast
    y_predict = model.predict(np.array(x_dataset))
    # Evaluation
    testScore = model.evaluate(np.array(x_dataset), np.array(y_dataset))
    print(f"Test Score Loss: {testScore[0]:0.4f}")
    # Plot Results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.plot(y_dataset)
    plt.plot(y_predict)
    plt.show()