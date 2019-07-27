# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:48:58 2019

@author: w
"""

import ffn
from pyetf.data import eod
import simdkalman
import numpy as np

# retrieve data from eod
etf_tickers = ['SPY']
prices = ffn.get(
        tickers=etf_tickers, market='US', 
        provider=eod, 
        start='2013-01-01')

kf = simdkalman.KalmanFilter(
    state_transition = [[1,1],[0,1]],        # matrix A
    process_noise = np.diag([0.1, 0.01]),    # Q
    observation_model = np.array([[1,0]]),   # H
    observation_noise = 1.0)                 # R

# smooth and explain existing data
smoothed = kf.smooth(prices)

