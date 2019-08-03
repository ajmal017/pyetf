# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:59:34 2019
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, GRU

def train_model(x_train, y_train):
    return lstm_model_1(x_train, y_train)

def addFeatures(dataset):
    return addFeatures_2(dataset)

def addTarget(dataset):
    return addTarget_1(dataset)

def lstm_model_1(x_train, y_train):
    # 2. Build Model        
    # 2.1 setup model
    model = Sequential()
    model.add(LSTM(20, input_length=x_train.shape[1], input_dim=x_train.shape[2], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))    
    #model.add(Dropout(0.2))
    #model.add(Activation("relu"))
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()
    # 2.2 train model
    model.fit(x_train, y_train, epochs=100, batch_size=min(1000,x_train.shape[0]), verbose=2)
    return model

def gru_model_1(x_train, y_train):
    # 2. Build Model        
    # 2.1 setup model
    model = Sequential()
    model.add(GRU(20, input_length=x_train.shape[1], input_dim=x_train.shape[2], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))    
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()
    # 2.2 train model
    model.fit(x_train, y_train, epochs=1000, batch_size=min(1000,x_train.shape[0]), verbose=2)
    return model

# add features variance data as X
from pyetf.algos import diffMA, slopeMA, addVAR
def addFeatures_1(dataset):
    dataset['r'] = dataset.pct_change() * 100
    dataset['weekday'] = dataset.index.weekday / 4
    dataset['diff'] = diffMA(dataset.price)
    dataset['slope'] = slopeMA(dataset.price)
    #dataset['historical_var'] = addVAR(dataset.price)
    #dataset['garch'] = addGARCH(dataset.price) # checked at 7.21 and improvement is limited.
    '''
    for e in dataset.columns:
        print(f"{e}: ({dataset[e].min():0.4f} : {dataset[e].max():0.4f})")
    '''
    return dataset.dropna()

def addFeatures_2(dataset):
    dataset['diff'] = diffMA(dataset.price)
    dataset['slope'] = slopeMA(dataset.price)
    dataset['historical_var'] = addVAR(dataset.price)
    return dataset.dropna()

def addFeatures_3(dataset):
    #dataset['r'] = dataset.pct_change() * 100
    dataset['historical_var'] = addVAR(dataset.price)
    return dataset.dropna()

from pyetf.algos import addCOV
def addFeatures_two_1(dataset):
    #dataset['r1'] = dataset[dataset.columns[0]].pct_change() * 100
    #dataset['r2'] = dataset[dataset.columns[1]].pct_change() * 100
    #dataset['weekday'] = dataset.index.weekday / 4
    dataset['historical_covar'] = addCOV(dataset[dataset.columns[0]], dataset[dataset.columns[1]])
    return dataset #dropna() at process data

# add forecast variance data as Y
from pyetf.algos import future_mean_var
def addTarget_1(dataset, futureDays=30):
    prices = dataset.price
    m = futureDays
    y_var = []
    for t in range(0, len(prices)-m+1):
        p = prices.iloc[t:t+m]
        _, var = future_mean_var(p)
        y_var.append(var*10000)
    '''
    # normalization
    y_min = min(y_var)
    y_max = max(y_var)
    y_distance = y_max-y_min
    for t in range(len(y_var)):
        y_var[t] = (y_var[t]-y_min)/y_distance
    '''
    for t in range(len(y_var), len(prices)):
        y_var.append(np.nan)
    dataset['y'] = y_var
    return dataset.dropna()

def addTarget_1_minus(dataset, futureDays=30):
    prices = dataset.price
    m = futureDays
    y_var = []
    for t in range(0, len(prices)-m+1):
        p = prices.iloc[t:t+m]
        _, var = future_mean_var(p, True)
        y_var.append(var*10000)
    for t in range(len(y_var), len(prices)):
        y_var.append(np.nan)
    dataset['y'] = y_var
    return dataset.dropna()

def addTarget_2(dataset, futureDays=30, sm=5):
    prices = dataset.price
    m = futureDays
    y_var = []
    for t in range(0, len(prices)-m+1):
        p = prices.iloc[t:t+m]
        _, var = future_mean_var(p)#, True)
        y_var.append(var*10000)
        
    ym_var = []
    for t in range(sm, len(y_var)):
        ym_var.append(sum(y_var[t-sm:t])/sm)

    for t in range(len(ym_var), len(prices)):
        ym_var.append(np.nan)
    dataset['y'] = ym_var
    return dataset.dropna()

from pyetf.algos import forecast_var_from_constant_mean
def addTarget_3(dataset, futureDays=1, hln=200):
    prices = dataset.price
    m = futureDays
    y_var = []
    for t in range(hln, len(prices)-m+1):
        p = prices.iloc[t-hln:t+m]
        var, _ = forecast_var_from_constant_mean(p.to_returns().dropna())
        y_var.append(var*10000)
    y_var = np.append(np.zeros([len(prices)-len(y_var),1]), y_var)
    dataset['y'] = y_var
    return dataset.dropna()

# add forecast variance data as Y
# checked at 07.20, forecast mean is more difficult than f- volitility
def addTarget_mean(dataset, futureDays=30):
    prices = dataset.price
    m = futureDays
    y_m = []
    for t in range(0, len(prices)-m+1):
        p = prices.iloc[t:t+m]
        mean, _ = future_mean_var(p)
        y_m.append(mean*100)
    for t in range(len(prices)-m+1, len(prices)):
        y_m.append(np.nan)
    dataset['y'] = y_m
    print(max(y_m), min(y_m))
    return dataset.dropna()

def addTarget_two_1(dataset, hln=170, futureDays=30):
    ts1 = dataset[dataset.columns[0]].to_returns().dropna()
    ts2 = dataset[dataset.columns[1]].to_returns().dropna()
    cov = []
    #cov.append(np.nan) # add 1 when dropna at prices->returns 
    for t in range(hln):
        cov.append(np.nan)
    for t in range(hln, len(ts1)-futureDays):        
        f_cov = np.cov(ts1[t-hln:t+futureDays], ts2[t-hln:t+futureDays])
        cov.append(f_cov[0][1]*10000)
    for t in range(len(cov), len(ts1)+1):
        cov.append(np.nan)
    dataset['y'] = cov
    return dataset