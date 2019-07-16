# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:07:12 2019

@author: w
"""

import __init__
import numpy as np
import os
from pyetf.algos import addFeatures, addTarget, buildXY, normalise_windows

# 1. Data Process
import ffn
from pyetf.data import eod
# 1.1 read data
etf_tickers = ['SPY']
model_path = os.getcwd() + "\\examples\\keras_model\\"
model_filename = model_path + 'est_var(' + etf_tickers[0].lower() + ').h5'
start_date_str = '2013-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)
dataset = prices.copy()
dataset = dataset.rename({etf_tickers[0].lower():'price'}, axis=1)
# 1.2 add features to X
dataset = addFeatures(dataset)
# 1.3 add targets to Y
dataset = addTarget(dataset)
# 1.4 structure train and test data
dataset = dataset.drop(columns='price')
x_dataset, y_dataset = buildXY(dataset)
# 1.5 normalization
#x_dataset = normalise_windows(x_dataset)

# 3. Prediction and Evaluation
from keras.models import load_model
# 3.1 load model
model_load = load_model(model_filename)
# 3.2 predication
y_predict = model_load.predict(np.array(x_dataset))
# 3.3 evaluation
testScore = model_load.evaluate(np.array(x_dataset), np.array(y_dataset))
print(f"Test Score Loss: {testScore[0]:0.4f}")

# 4. Plot Results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.plot(y_dataset)
plt.plot(y_predict)
plt.show()