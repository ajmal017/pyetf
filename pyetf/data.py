# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pandas.compat import StringIO
from pandas.core.base import PandasObject
import requests
import tushare as ts
from ffn.utils import memoize
from ffn.core import PerformanceStats, GroupStats
from pyetf.utils import format_date, sanitize_dates
    
def extend_ffn():
    """
    Extends ffn function, i.e. display items
    Ex:
        prices.to_returns().dropna().calc_annual_return()
        (where prices would be a DataFrame)
    """
    PerformanceStats._stats = _stats
    GroupStats._stats = _stats
    PandasObject.calc_annual_return = calc_annual_return
    PandasObject.calc_annual_std = calc_annual_std
    PandasObject.calc_sharpe_ratio = calc_sharpe_ratio
    PandasObject.to_weekly = to_weekly
    """
    Other useful functions defined in ffn:
        PandasObject.to_returns = to_returns
        PandasObject.to_log_returns = to_log_returns
        PandasObject.rebase = rebase        
        PandasObject.to_monthly = to_monthly    
        PandasObject.to_drawdown_series = to_drawdown_series
        PandasObject.calc_max_drawdown = calc_max_drawdown
        PandasObject.calc_total_return = calc_total_return
        PandasObject.calc_stats = calc_stats
        PandasObject.calc_mean_var_weights = calc_mean_var_weights
    """
            
def _stats(self):
        stats = [('start', 'Start', 'dt'),
                 ('end', 'End', 'dt'),
                 (None, None, None),
                 ('total_return', 'Total Return', 'p'),
                 ('daily_mean', 'Mean (ann.)', 'p'),
                 ('daily_vol', 'Vol (ann.)', 'p'),
                 ('daily_sharpe', 'Sharpe Ratio', 'n'),
                 ('max_drawdown', 'Max Drawdown', 'p')]
        return stats
        
def to_weekly(series, method='ffill', how='end'):
    """
    Convenience method that wraps asfreq_actual
    with 'M' param (method='ffill', how='end').
    """
    return series.asfreq_actual('W', method=method, how=how)

def calc_annual_return(r, frac=252):
    """
    frac 
        Daily : 252
        Weekly: 52
        Monthly: 12
        default is Daily
    """
    return r.mean() * frac

def calc_annual_std(r, frac=252):
    return np.std(r, ddof=1) * np.sqrt(frac)

def calc_sharpe_ratio(r, rf=0., frac=252):
    m = calc_annual_return(r, frac)
    s = calc_annual_std(r, frac)
    sr = (m - rf) / s
    return sr.fillna(0.)
    
@memoize
def eod(ticker="AAPL", market="US", 
        field=None, mrefresh=False, existing=None, 
        start=None, end=None):
    """    
    Helper function for retrieving data as a DataFrame.
    Data provider is eod.
    Args:
        * ticker (list, string, csv string): Ticker to download.
        * market (string): exchange market i.e. "US", "HK".        
        * field (string): if default, field is "Adjusted_close"
        * mrefresh (bool): if True, then Ignore memoization.
        * existing (DataFrame): Existing DataFrame to append returns
            to - used when we download from multiple sources
        * start (string, number): retrieve data since start date 
            "2010-01-01" or 2010
        * end (string, number): retrieve data until end date
            if None, retrieve until latest date 
    """
    api_token = "***"
    session = None
    if field is None:
        field = "Adjusted_close"
    if session is None:
        session = requests.Session()       
        start, end = sanitize_dates(start, end)
        ticker = "".join([ticker,'.',market])
        url = "https://eodhistoricaldata.com/api/eod/%s" % ticker     
        params = {        
            "api_token": api_token,
            "from": format_date(start),
            "to": format_date(end)
        }
        r = session.get(url, params=params)
        if r.status_code == requests.codes.ok:
            df = pd.read_csv(
                    StringIO(r.text), 
                    skipfooter=1, 
                    parse_dates=[0], 
                    index_col=0, 
                    engine="python"
                    )
            p=df[field]
            return p
    else:
        raise Exception(r.status_code, r.reason, url)
        
@memoize
def tushare(ticker="510310", market="SH", asset="FD",
        field=None, mrefresh=False, existing=None, 
        start=None, end=None):
    """    
    Helper function for retrieving data as a DataFrame.
    Data provider is tushare pro.
    Args:
        * ticker (list, string, csv string): Ticker to download.
        * market (string): exchange market i.e. "SH", "SZ".
        * asset (string): E stock, I index, C e-coin, FT future,
            FD fund, O option, default is FD          
        * field (string): if default, field is "Adjusted_close"
        * mrefresh (bool): if True, then Ignore memoization.
        * existing (DataFrame): Existing DataFrame to append returns
            to - used when we download from multiple sources
        * start (string, number): retrieve data since start date 
            "2010-01-01" or 2010
        * end (string, number): retrieve data until end date
            if None, retrieve until latest date 
    """
    api_token = "***"
    if field is None:
        field = "close"
    ts.set_token(api_token)
    start, end = sanitize_dates(start, end)
    if start is not None: start = format_date(start).replace("-","")
    if end is not None: end = format_date(end).replace("-","")
    ticker = "".join([ticker,'.',market])
    df = ts.pro_bar(
            ts_code=ticker,
            asset=asset,
            start_date=start,
            end_date=end
            )
    df['Date'] = pd.to_datetime(df['trade_date'])
    df.set_index("Date", inplace=True)
    p=df[field].sort_index()
    return p