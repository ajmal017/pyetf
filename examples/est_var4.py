# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:44:49 2019

@author: w
"""

import __init__
import numpy as np
import pandas as pd
import ffn
from pyetf.data import eod
from pyetf.figure import plot_chart

# retrieve data from eod and combine
etf_tickers = ['IVV', 'TLT', 'IEF', 'GLD', 'DBC']; market='US'
etf_tickers = ['2800']; market='HK'
benchmark = etf_tickers[0].lower()

# retrieve data from eod and combine
start_year = 2013; end_year=2019; end_date=str(end_year)+'-06-30'
start_date_str = str(start_year-1) + '-01-01'
prices = ffn.get(tickers=etf_tickers, market=market, 
                 provider=eod, 
                 start=start_date_str)
pl_tmp = prices.loc[prices.index.year>=start_year].copy()
pl = pl_tmp.loc[pl_tmp.index<=end_date].copy()

# Historical
var = []
covar = []
m = hist_length = 200 
for t in range(0, len(prices)-m+1):
    p = prices.iloc[t:t+m].copy()   
    r = p.to_returns().dropna()
    var.append(r.var().values)
    covar.append(r.cov().values)
# pandas dataframe
pdvar = pd.DataFrame(data=var, index=prices.index[len(prices)-len(var):len(prices)], columns=prices.columns)
var_tmp = pdvar.loc[pdvar.index.year>=start_year].copy()
var = var_tmp.loc[var_tmp.index<=end_date].copy()

# Mean_Variance
mv = []
for t in range(0, len(prices)-m+1):
    p = prices.iloc[t:t+m].copy()   
    r = p.to_returns().dropna()
    v = []
    for e in r.columns:
        v_tmp = -r[e].var()*2+r[e].mean()
        v.append(v_tmp)
    mv.append(v)
pdmv = pd.DataFrame(data=mv, index=prices.index[len(prices)-len(mv):len(prices)], columns=prices.columns)
var_tmp = pdmv.loc[pdmv.index.year>=start_year].copy()
mv = var_tmp.loc[var_tmp.index<=end_date].copy() 
mv_t = mv.copy()
uplevel=0.0005
for e in mv.columns:
    for t in range(1, len(mv[e])):
        if mv_t[e][t-1]<0:
            if mv[e][t]<mv_t[e][t-1]:
                mv_t[e][t]=mv[e][t]
            elif mv[e][t]<uplevel:
                mv_t[e][t]=mv_t[e][t-1]
            
'''
# LSTM    
from pyetf.algos import load_keras_model, processData
from pyetf.keras_model import addFeatures, addTarget
var_lstm = []
for e in prices.columns:
    # Initializing Data and Load Model
    dataset, model = load_keras_model(prices[e])
    # Process Dataset
    x_dataset, y_dataset = processData(addFeatures, addTarget, dataset)
    # Forecast
    y_predict = model.predict(np.array(x_dataset))
    var_lstm.append(y_predict)
# pandas dataframe
pdvar_lstm = pd.DataFrame(data=var_lstm[0], index=prices.index[len(prices)-len(var_lstm[0]):len(prices)])
for i in range(1,len(prices.columns)):
    pdvar_lstm[i] = var_lstm[i]
pdvar_lstm.columns = prices.columns
var_tmp = pdvar_lstm.loc[pdvar_lstm.index.year>=start_year].copy()
var_lstm = var_tmp.loc[var_tmp.index<=end_date].copy() # half end year

# GARCH
from pyetf.algos import forecast_var_from_garch
var_garch = []
for t in range(0, len(prices)-m+1):
    p = prices.iloc[t:t+m].copy()   
    r = p.to_returns().dropna()
    v = []
    for i in range(len(r.columns)):
        v_tmp, _ = forecast_var_from_garch(100.0*r[r.columns[i]])
        v.append(v_tmp)
    var_garch.append(v)
pdvar_garch = pd.DataFrame(data=var_garch, index=prices.index[len(prices)-len(var_garch):len(prices)], columns=prices.columns)
var_tmp = pdvar_garch.loc[pdvar_garch.index.year>=start_year].copy()
var_garch = var_tmp.loc[var_tmp.index<=end_date].copy() # half end year
'''
# plot
e=benchmark
varl = pd.DataFrame(data=var[e], index=var.index)
#varl['lstm']=var_lstm[e]/10000
#varl['garch']=var_garch[e]/10000
varl['risk']=mv[e]
#varl['mv']=varl[e]*(1+mv_t[e]*10000)
#varl['mv'].loc[varl['mv']<=0]=0.00001
plot_chart(pl[e], sub_fill=varl[['risk']])
#plot_chart(pl[e], sub=varl)
