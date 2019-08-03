# -*- coding: utf-8 -*-

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from decorator import decorator

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

# garch
def addGARCH(ds, hln=200):
    ts = 100*ds.to_returns().dropna()
    hts = ts[:hln].values
    var = []
    # rolling estimate var
    while (len(hts)<len(ts)):
        f_var, _ =  forecast_var_from_garch(hts[-hln:])
        var.append(f_var)
        hts = np.append(hts, ts.iloc[len(hts)])
    print(max(var), min(var))
    var = np.append(np.zeros([len(ds)-len(var),1]), var)
    return var

# historical var
def addVAR(ds, hln=200):
    ts = 100*ds.to_returns().dropna()
    hts = ts[:hln].values
    var = []
    # rolling estimate var
    while (len(hts)<len(ts)):
        f_var, _ =  forecast_var_from_constant_mean(hts[-hln:])
        var.append(f_var)
        hts = np.append(hts, ts.iloc[len(hts)])
    #print(max(var), min(var))
    var = np.append(np.zeros([len(ds)-len(var),1]), var)
    return var

# historical cov
def addCOV(ds1, ds2, hln=200):
    ts1 = ds1.to_returns().dropna().values
    ts2 = ds2.to_returns().dropna().values
    cov = []
    #cov.append(np.nan) # add 1 when dropna at prices->returns 
    for t in range(hln):
        cov.append(np.nan)
    for t in range(hln, len(ts1)+1):
        f_cov = np.cov(ts1[t-hln:t], ts2[t-hln:t])
        cov.append(f_cov[0][1]*10000)
    return cov

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

from arch.univariate import ConstantMean
@forecast_var
def forecast_var_from_constant_mean(returns):
    """
    returns is historical returns
    """    
    return ConstantMean(returns)

from arch import arch_model
@forecast_var
def forecast_var_from_garch(returns):
    """
    returns is historical returns
    """    
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
        if len(dr) == 0:
            dr.append(0.)
    else:
        for d in range(1,m):
            dr.append((p[d]/p[0])**(1/d)-1)
    mean = np.mean(dr)
    var = np.var(dr)
    return mean, var

# future mean and var
def future_covar(p1, p2=None):
    """
    p1 and p2 are numpy and prices series in future fm(30) dates
    + historical hm(200-fm) dates
    p1 = p2: calculate var
    """
    r1 = np.diff(p1)/p1[0:len(p1)-1]
    if p2 is None:        
        return np.var(r1)
    else:
        r2 = np.diff(p2)/p1[0:len(p2)-1]
        return np.cov(r1, r2)

# under keras model scheme
def strucutre_keras_model(train_model, addFeatures, addTarget, prices, prices_two=None, model_path="\\keras_model\\"):
    """
    * prices: pandas series (or dataframe) with date index and prices
    * function will save model estimated by keras 
    to a h5 file named 'est_var(_ticker_).h5'
    * load model
    from keras.models import load_model 
    model_load = load_model('est_var(_ticker_).h5')
    """
    # 1. Data Process
    if prices_two is None:
    # 1.1 initial data    
        dataset, model_filename = initData(prices, model_path)
    # 1.2 process data
        x_dataset, y_dataset = processData(addFeatures, addTarget, dataset)
    else:
        dataset, model_filename = initData_two(prices, prices_two, model_path)
        x_dataset, y_dataset = processData_two(addFeatures, addTarget, dataset)
    # 1.3 split train set and test set
    x_train, y_train, x_test, y_test = splitDataset(x_dataset, y_dataset)
    # 1.4 shuttle train set
    x_train, y_train = shuffleDataset(x_train, y_train)
    # 2. Build Model        
    # 2.1 setup model    
    # 2.2 train model
    model = train_model(x_train, y_train)
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

# stucture X from dataset to forecast
def buildX(dataset, pastDays=30):
    """
    Result -> numpy
    """
    m = pastDays
    x_dataset = dataset.values
    dataX = []
    for t in range(0, len(dataset)-m+1):
        dataX.append(x_dataset[t:(t+m)])
    return np.array(dataX)

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

# initial Data and model name
def initData(prices, model_path, model_name='est_var'):    
    if isinstance(prices, pd.core.series.Series):
        e = prices.name
        dataset = pd.DataFrame(prices)
    else:
        e = prices.columns[0]
        dataset = prices.copy()
    print(f"{e}")
    dataset = dataset.rename({e:'price'}, axis=1)
    model_path = os.getcwd() + model_path
    model_filename = model_path + model_name + '(' + e + ').h5'
    return dataset, model_filename

# initial Data and model name
def initData_two(prices_one, prices_two, model_path, model_name='est_cov'):    
    if isinstance(prices_one, pd.core.series.Series):
        e1 = prices_one.name
        dataset = pd.DataFrame(prices_one)
    else:
        e1 = prices_one.columns[0]
        dataset = prices_one.copy()
    dataset = dataset.rename({e1:'price_one'}, axis=1)
    if isinstance(prices_two, pd.core.series.Series):
        e2 = prices_two.name
        dataset[e2] = pd.DataFrame(prices_two)
    else:
        e2 = prices_two.columns[0]
        dataset[e2] = prices_two.columns[0]          
    dataset = dataset.rename({e2:'price_two'}, axis=1)
    print(f"{e1} {e2}")
    model_path = os.getcwd() + model_path
    model_filename = model_path + model_name + '(' + e1+'_'+e2 + ').h5'
    return dataset, model_filename
    
# process data: add features and add Y
def processData(addFeatures, addTarget, dataset):
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

# process data: add features and add Y
def processData_two(addFeatures, addTarget, dataset, pastDays=30):    
    # 1.2 add features to X
    dataset = addFeatures(dataset)
    # 1.3 add targets to Y
    dataset = addTarget(dataset)
    # 1.4 structure train and test data
    dataset = dataset.dropna()
    dataset = dataset.drop(columns='price_one')
    dataset = dataset.drop(columns='price_two')
    #print(dataset.head())
    #print(dataset.tail())
    x_dataset, y_dataset = buildXY(dataset, pastDays)
    # 1.5 normalization
    #x_dataset = normalise_windows(x_dataset)  
    return x_dataset, y_dataset  

# lstm var
from time import process_time
def forecast_var_from_lstm(addFeatures, prices, model_path="\\keras_model\\"):
    """
    Prices is one asset's price data, in either DataFrame or Pandas Series
    """
    # Initializing Data and Load Model
    start_time = process_time()
    dataset, model = load_keras_model(prices)
    print(f"load data and model: {process_time()-start_time:0.4f}s")
    start_time = process_time()
    dataset = addFeatures(dataset)
    x_dataset = dataset.drop(columns='price')
    x_dataset = buildX(x_dataset)
    print(f"process dataset: {process_time()-start_time:0.4f}s")
    start_time = process_time()
    f_var = np.append(np.zeros([len(prices)-len(x_dataset),1]), model.predict(np.array(x_dataset)))
    print(f"calc var: {process_time()-start_time:0.4f}s")
    return f_var

#lstm cov
def forecast_cov_from_lstm(addFeatures, prices_one, prices_two, pastDays=30, model_path="\\keras_model\\"):
    """
    Prices is one asset's price data, in either DataFrame or Pandas Series
    """
    # Initializing Data and Load Model
    start_time = process_time()
    dataset, model_filename = initData_two(prices_one, prices_two, model_path)
    model = load_model(model_filename)
    print(f"load data and model: {process_time()-start_time:0.4f}s")
    start_time = process_time()
    dataset = addFeatures(dataset)
    dataset = dataset.drop(columns='price_one')
    dataset = dataset.drop(columns='price_two')
    x_dataset = buildX(dataset, pastDays)
    print(f"process dataset: {process_time()-start_time:0.4f}s")
    start_time = process_time()
    f_cov = np.append(np.zeros([len(prices_one)-len(x_dataset),1]), model.predict(np.array(x_dataset)))
    print(f"calc cov: {process_time()-start_time:0.4f}s")
    return f_cov.tolist()