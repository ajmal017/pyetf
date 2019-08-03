# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:14:10 2019
All Weather Strategy
@author: w
"""
import __init__
import ffn
from pyetf.data import eod
from pyetf.alloc import rpw_mean_var, rpw_ledoit_wolf, rpw_lstm, rpw_standard, rpw_garch, rpw_future
from pyetf.alloc import to_weights
from pyetf.figure import plot_chart

# retrieve data from eod and combine
etf_tickers = ['IVV', 'GLD', 'VNQ', 'AGG', 'TLT', 'IEF', 'IEI', 'OIL', 'USO', 'DBC']
etf_tickers = ['IVV', 'TLT', 'IEF', 'GLD', 'DBC']
benchmark = etf_tickers[0].lower()
mc_budget = [0.30, 0.40, 0.15, 0.08, 0.07]

# retrieve data from eod and combine
start_year = 2013; end_year=2019; end_date=str(end_year)+'-06-30'
start_date_str = str(start_year-1) + '-01-01'
prices = ffn.get(tickers=etf_tickers, market='US', 
                 provider=eod, 
                 start=start_date_str)

minn = len(mc_budget)
maxn = minn+4

method = 'lstm'

# calc portfolio weights
if method == 'lstm':
    #lstm
    w = prices.to_weights(
            func_weighting=rpw_lstm, 
            model="lstm", 
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()
elif method == 'garch':
    #garch
    w = prices.to_weights(
            func_weighting=rpw_garch, 
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()
elif method == 'future':
    #future
    w = prices.to_weights(
            func_weighting=rpw_future,
            hist_length=-30,
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()    
elif method == 'standard':
    #historical
    w = prices.to_weights(
            func_weighting=rpw_standard,
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()
elif method == 'wolf':
    #historical
    w = prices.to_weights(
            func_weighting=rpw_ledoit_wolf,
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()
elif method == 'mean_var':
    #historical
    w = prices.to_weights(
            func_weighting=rpw_mean_var,
            model="mean_var", 
            risk_weights=mc_budget,
            min_assets_number = minn,
            max_assets_number = maxn
            ).dropna()
    
# calc portfolio performance
pf = prices.to_NAV(w)   

# plot portfolio
#pl = pf.copy() # all data
pl_tmp = pf.loc[pf.index.year>=start_year].copy()
pl = pl_tmp.loc[pl_tmp.index<=end_date].copy() # half end year
#pl = pl_tmp.loc[pl_tmp.index.year<=end_year].copy() # whole end year
pl.dropna()
w_tmp = w.loc[w.index.year>=start_year].copy()
wl = w_tmp.loc[w_tmp.index<=end_date].copy() # half end year
#wl = w_tmp.loc[w_tmp.index.year<=end_year].copy() # whole end year
wl.dropna()
pl = pl.rebase()
plot_chart(pl[['NAV', benchmark]], sub_fill=wl)
# performance
pl.calc_stats().display()
#for y in range(start_year, end_year+1):
#    pl[str(y)].calc_stats().display()