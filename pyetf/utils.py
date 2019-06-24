# -*- coding: utf-8 -*-

import pandas as pd
from pandas.api.types import is_number
import datetime

def format_date(dt):
    """
    Returns formated date
    """
    if dt is None:
        return dt
    return dt.strftime("%Y-%m-%d")

def sanitize_dates(start, end):
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