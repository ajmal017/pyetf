# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:50:21 2019

@author: w
"""
import ffn
from ffn import utils
import pandas as pd
from pandas.compat import StringIO
from pandas.api.types import is_number
import requests
import datetime

def _format_date(dt):
    """
    Returns formated date
    """
    if dt is None:
        return dt
    return dt.strftime("%Y-%m-%d")

def _sanitize_dates(start, end):
    """
    Return (datetime_start, datetime_end) tuple
    """
    if is_number(start):
        # regard int as year
        start = datetime.datetime(start, 1, 1)
    start = pd.to_datetime(start)
    if is_number(end):
        # regard int as year
        end = datetime.datetime(end, 1, 1)
    end = pd.to_datetime(end)
    if start is not None and end is not None:
        if start > end:
            raise Exception("end must be after start")
    return start, end

@utils.memoize
def eod(ticker="AAPL", market="US", field=None, mrefresh=False, start=None, end=None):
    """
    Data provider eod from 
    https://eodhistoricaldata.com/api/eod/AAPL.US?api_token={your_api_key}
    Provides memoization.
    """
    api_token="5cebe613e02010.08022109"
    session=None
    field="Adjusted_close"
    if session is None:
        session = requests.Session()       
        start, end = _sanitize_dates(start, end)
        ticker = "".join([ticker,'.',market])
        url = "https://eodhistoricaldata.com/api/eod/%s" % ticker      
        params = {        
            "api_token": api_token,
            "from": _format_date(start),
            "to": _format_date(end)
        }
        r = session.get(url, params=params)
        if r.status_code == requests.codes.ok:
            df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine="python")
            p=df[field]
            return p
    else:
        raise Exception(r.status_code, r.reason, url)