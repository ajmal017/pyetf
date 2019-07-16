# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:17:16 2019

@author: w
"""
import __init__
import ffn
from pyetf.data import eod
from pyetf.algos import MA, liner_regression, future_mean_var
import pandas as pd
import numpy as np

# retrieve data from eod
etf_tickers = ['2800']
prices = ffn.get(
        tickers=etf_tickers, market='HK', 
        provider=eod, 
        start='2005-01-01')
m=30
mean=[]
std=[]

for t in range(0,len(prices)-m+1):
    p = prices[t:t+m].values
    mean_t, var_t = future_mean_var(p, False)
    mean.append(mean_t)
    std.append(var_t**0.5)
    
p = prices['2800'][0:len(mean)].values
df = pd.DataFrame({'2800':p,'mean':mean,'std':std}, index=prices.index[0:len(mean)])
df['mal'] = MA(df['2800'],120)
df['mas'] = MA(df['2800'],10)
df['diff'] = (df['mas']/df['mal']-1)/10
df['slope'] = df['diff'].pct_change()
df['mean_std'] = -2*df['std']+df['mean']
df['zero'] = 0.
df = df.dropna()

from pyetf.figure import plot_chart
plot_chart(
        df[['2800','mal']], 
        sub=df[['zero', 'mean']],
        sub_fill=df['std'])

#
df1= df[df['slope']>0]
df2= df[df['slope']<=0]
x1 = df1['diff'].values
y1 = -df1['mean_std'].values
b1,t1=liner_regression(x1,y1)
x2 = df2['diff'].values
y2 = -df2['mean_std'].values
b2,t2=liner_regression(x2,y2)