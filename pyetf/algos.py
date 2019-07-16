# -*- coding: utf-8 -*-

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Moving Average  
def MA(ds, n):  
    MA = pd.Series(ds.rolling(n).mean(), name = 'MA_' + str(n))   
    return MA

# difference between short MA and long MA
def diffMA(ds, l=60, s=5):
    """
    ds: dataset is pandas data series
    """
    ma_l = ds.rolling(l, min_periods=l).mean()
    ma_s = ds.rolling(s, min_periods=s).mean()
    return (ma_s/ma_l)-1

# Linear Regression
import statsmodels.formula.api as smf
def liner_regression(x,y):
    model = smf.OLS(y,x)
    results = model.fit()
    b = results.params
    R = results.rsquared
    pvalue = results.pvalues
    t='Y=%0.4fX --- R2=%0.2f%% --- p-value=%0.4f' %(b[0], R*100, pvalue[0])
    return b,t

# slope of MA
def slopeMA(ds, m=60, dw=5):
    ma = ds.rolling(m, min_periods=1).mean()
    slope = ma.copy()
    x = np.arange(1,dw+1)/100.0
    for t in range(dw,len(slope)):
        y = ma[t-dw+1:t+1] / ma[t-dw+1:t+1].mean() - 1           
        slope[t], _ = liner_regression(x,y)
    return slope

# Seek Best Garch Model
import statsmodels.tsa.api as smt
def seek_garch_model(TS):
    """
    TS is returns of a price-series
    numpy array or array
    # Seek Best GARCH Model
    res_tup = seek_garch_model(ts)
    order = res_tup[1]
    p_ = order[0]
    o_ = order[1]
    q_ = order[2]
    # Using student T distribution usually provides better fit
    am = arch_model(ts, p=p_, o=o_, q=q_, dist='StudentsT')
    res = am.fit(update_freq=5, disp='off')
    fig = res.plot(annualize='D')
    print(res.summary())
    ts_plot(res.resid, lags=30)
    """
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i,d,j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl

#under arch model scheme
from decorator import decorator
@decorator
def forecast_var(model_est_var, *args, **kwargs):                     
    """
    Use historical data (0 to t) to forecast variance at t+1
    via the model (defined in arch)

    Args:
        * args[0]: returns (numpy array or array): Returns for security.

    Returns:
        forecast variance: float
        residuals: array
    """
    if len(args)<1:
        raise Exception("Not Enough Parameters")
    
    m = model_est_var(*args, **kwargs)
    res = m.fit(update_freq=5, disp='off')
    return res.forecast().variance.values[-1][0], res.resid

@forecast_var
def forecast_var_from_constant_mean(returns):
    """
    returns is historical returns
    """
    from arch.univariate import ConstantMean
    return ConstantMean(returns)

@forecast_var
def forecast_var_from_garch(returns):
    """
    returns is historical returns
    """
    from arch import arch_model
    return arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')

@forecast_var
def forecast_var_from_best(returns):
    """
    returns is historical returns
    """
    from pyetf.algos import seek_garch_model
    from arch import arch_model
    res_tup = seek_garch_model(returns)
    order = res_tup[1]
    p_ = order[0]
    o_ = order[1]
    q_ = order[2]
    return arch_model(returns, p=p_, o=o_, q=q_, dist='StudentsT')

# future mean and var
def future_mean_var(p, negative=False):
    """
    p is numpy and prices series in future m dates
    negative is True:   calculate if p(t) < p(0)
    negative is False:  calculate all p(t)
    """
    m = len(p)
    dr = []
    if negative:
        for d in range(1,m):  
            if p[d]<p[0]:
                dr.append((p[d]/p[0])**(1/d)-1)
    else:
        for d in range(1,m):
            dr.append((p[d]/p[0])**(1/d)-1)
    mean = np.mean(dr)
    var = np.var(dr)
    return mean, var

# under keras model scheme
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
def strucutre_keras_model(prices, model_path="\\keras_model\\"):
    """
    * prices: pandas series (or dataframe) with date index and prices
    * function will save model estimated by keras 
    to a h5 file named 'est_var(_ticker_).h5'
    * load model
    from keras.models import load_model 
    model_load = load_model('est_var(_ticker_).h5')
    """
    # 1. Data Process
    # 1.1 initial data
    dataset, model_filename = initData(prices, model_path)    
    # 1.2 process data
    x_dataset, y_dataset = processData(dataset)
    # 1.3 split train set and test set
    x_train, y_train, x_test, y_test = splitDataset(x_dataset, y_dataset)
    # 1.4 shuttle train set
    x_train, y_train = shuffleDataset(x_train, y_train)
    # 2. Build Model        
    # 2.1 setup model
    model = Sequential()
    model.add(LSTM(20, input_length=x_train.shape[1], input_dim=x_train.shape[2]))
    model.add(Dense(1))
    model.add(Dropout(0.2))
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
    model.summary()
    # 2.2 train model
    model.fit(x_train, y_train, epochs=1000, batch_size=min(1000,x_train.shape[0]), verbose=1)
    # 2.3 save model
    model.save(model_filename)
    # 3 evaluation
    trainScore = model.evaluate(x_train, y_train)
    testScore = model.evaluate(x_test, y_test)
    print(f"Train Score Loss: {trainScore[0]:0.4f}")
    print(f"Test Score Loss: {testScore[0]:0.4f}")
    
    # 4. Plot Results    
    plt.figure(figsize=(10, 8))
    #plt.plot(y_dataset)
    #plt.plot(y_predict)
    plt.plot(y_test)
    plt.plot(model.predict(x_test))
    plt.show()

from keras.models import load_model
def load_keras_model(prices, model_path="\\keras_model\\"):
     # 1. Data Process
    # 1.1 initial data
    dataset, model_filename = initData(prices, model_path)
    model = load_model(model_filename)
    return dataset, model
    
def addFeatures(dataset):
    dataset['r'] = dataset.pct_change() * 100
    dataset['weekday'] = dataset.index.weekday / 4
    dataset['diff'] = diffMA(dataset.price)
    dataset['slope'] = slopeMA(dataset.price)
    for e in dataset.columns:
        print(f"{e}: ({dataset[e].min():0.4f} : {dataset[e].max():0.4f})")
    return dataset.dropna()

# add forecast variance data as Y
def addTarget(dataset, futureDays=30):
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
    for t in range(len(prices)-m+1, len(prices)):
        y_var.append(np.nan)
    dataset['y'] = y_var
    return dataset.dropna()

# stucture X and Y from dataset
def buildXY(dataset, pastDays=30):
    """
    Result -> numpy
    """
    m = pastDays
    x_dataset = dataset.drop(columns='y').values
    y_dataset = dataset['y'].values
    dataX, dataY = [], []
    for t in range(0, len(dataset)-m+1):
        dataX.append(x_dataset[t:(t+m)])
        dataY.append(y_dataset[t+m-1])
    return np.array(dataX), np.array(dataY)

# normalize dataset
from sklearn.preprocessing import MinMaxScaler
def normalise_windows(window_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalised_data = []
    for window in window_data:
        normalised_window = scaler.fit_transform(window)
        normalised_data.append(normalised_window)
    return normalised_data

# split dataset to train and test
def splitDataset(x_dataset, y_dataset, train_size_ratio=0.6):
    train_size = int(len(x_dataset) * train_size_ratio)
    x_train, x_test = x_dataset[0:train_size], x_dataset[train_size:len(x_dataset)]
    y_train, y_test = y_dataset[0:train_size], y_dataset[train_size:len(y_dataset)]
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# random train dataset 
def shuffleDataset(x, y):
    np.random.seed(10)
    randomList = np.arange(x.shape[0])
    np.random.shuffle(randomList)
    return x[randomList], y[randomList]

# initial Data
def initData(prices, model_path):    
    if isinstance(prices, pd.core.series.Series):
        e = prices.name
        dataset = pd.DataFrame(prices)
    else:
        e = prices.columns[0]
        dataset = prices.copy()
    print(f"{e}")
    dataset = dataset.rename({e:'price'}, axis=1)
    model_path = os.getcwd() + model_path
    model_filename = model_path + 'est_var(' + e + ').h5'
    return dataset, model_filename
    
# process data: add features and add Y
def processData(dataset):
    # 1.2 add features to X
    dataset = addFeatures(dataset)
    # 1.3 add targets to Y
    dataset = addTarget(dataset)
    # 1.4 structure train and test data
    dataset = dataset.drop(columns='price')
    x_dataset, y_dataset = buildXY(dataset)
    # 1.5 normalization
    #x_dataset = normalise_windows(x_dataset)  
    return x_dataset, y_dataset  

# lstm var
def forecast_var_from_lstm(prices, model_path="\\keras_model\\"):
    """
    Prices is one asset's price data, in either DataFrame or Pandas Series
    """
    # Initializing Data and Load Model
    dataset, model = load_keras_model(prices)
    dataset = addFeatures(dataset)
    x_dataset = dataset.drop(columns='price')
    f_var = model.predict(np.array(x_dataset[-2:-1]))
    return f_var[-1]
    