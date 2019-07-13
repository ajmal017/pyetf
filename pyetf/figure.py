# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

def ts_plot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 

def plot_chart(main, sub=None, sub_fill=None):
    """
    plot 
    main window: 
        x: datetime
        y: each column in main
    sub plot window (if not None):
        x: same as main
        y: each column in sub
    sub fill window (if not None):
        x: same as main
        y: each column in sub_fill
    """
    cols = 1
    rows = 1
    if sub is not None: rows+=1
    if sub_fill is not None: rows+=1    
    # Set figure and gridspec
    fig = set_fig(rows=rows)
    gs = set_gs(rows=rows, cols=cols)
    ax = []
    _h = 0
    # Set ax_main
    ax.append(fig.add_subplot(gs[_h, 0]))
    plt.setp(ax[_h].get_xticklabels(), rotation=0)
    # Set Axis     
    ax[_h] = set_ax_format(ax[_h], fig, tickers=set_datetime_ticker(main.index), ylabel='Price') 
    # Plot ax_main
    if isinstance(main, pd.core.series.Series):
        plot_y(ax[_h], main, label=main.name, linewidth=3)
    else:
        for i in range(len(main.columns)):
            if i==0:
                plot_y(ax[_h], main[main.columns[i]], label=main.columns[i], linewidth=2) 
            else:
                plot_y(ax[_h], main[main.columns[i]], label=main.columns[i])
    # Set ax_sub
    if sub is not None:
        _h+=1
        ax.append(fig.add_subplot(gs[_h, 0], sharex=ax[_h-1]))
        plt.setp(ax[_h-1].get_xticklabels(), visible=False)
        ax[_h] = set_ax_format(ax[_h], fig)
        if isinstance(sub, pd.core.series.Series):
            plot_y(ax[_h], sub, label=sub.name)
        else:
            for e in sub.columns:
                plot_y(ax[_h], sub[e], label=e)
    # Set ax_fill
    if sub_fill is not None:
        _h+=1
        ax.append(fig.add_subplot(gs[_h, 0], sharex=ax[_h-1]))
        plt.setp(ax[_h-1].get_xticklabels(), visible=False)
        ax[_h] = set_ax_format(ax[_h], fig)
        if isinstance(sub_fill, pd.core.series.Series):
            plot_stack(ax[_h], sub_fill, label=sub_fill.name)
        else:            
            plot_stack(ax[_h], sub_fill, label=sub_fill.columns)

def set_fig(w=-1, h=-1, rows=1, width_factor=3.0, height_factor=1, height_ratios=1.0):    
    """
    set figure size
    """
    if(w<0 or h<0):
        f_size = plt.gcf().get_size_inches()
        w = f_size[0]*width_factor
        h = f_size[1]*height_factor*(height_ratios+rows-1)
    fig = plt.figure(figsize=(w, h))
    fig.get_tight_layout()
    return fig

def set_gs(rows=1, cols=1, height_ratios=2.4):
    """
    set grid
    """
    if rows>1:
        hr = [height_ratios]
        for i in range(1,rows):
            hr.append(1)
    else:
        hr = None
    wr = None
    gs = gridspec.GridSpec(rows, cols, width_ratios=wr, height_ratios=hr, hspace=0)
    return gs

def set_ax_format(ax, fig, tickers=None, xlabel=None, ylabel=None):
    if tickers is not None:
        ax = set_xlimit(ax, tickers)    
        def format_date(x, pos=None):
            if x<0 or x>len(tickers)-1:
                return ''
            return tickers[int(x)]    
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_date))    
        ax = set_xlocator(ax, tickers, fig)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax

def _get_freq(w, l, u=50, s=5):
    '''
    w : figure pixel
    l : number of tickers
    u : one ticker in pixel
    s : search min l%freq via freq+s
    '''
    f = w//u
    d = l%f
    freq = f
    for i in range(s):
        if d > l%(f+i):
            d = l%(f+i)
            freq = f+i
    return freq

def set_xlocator(ax, tickers, fig):
    fig_width = fig.get_size_inches()[0] * fig.dpi
    num_tickers = len(tickers)
    freq = _get_freq(fig_width, num_tickers)
    if freq < num_tickers :
        ax.xaxis.set_major_locator(mticker.MaxNLocator(freq, prune='both'))
    return ax
    
def set_xlimit(ax, tickers):
    xax = np.arange(len(tickers))
    xmin = xax[0]-1
    xmax = xax[-1]+1
    ax.set_xlim(xmin, xmax)
    return ax

def set_datetime_format(dt):
    freq_minutes = np.diff(dt).min().astype(float)/1000000000/60/60
    if freq_minutes < 24:
        datetime_format = '%Y\n%m-%d\n%H-%M'
    elif freq_minutes < 672:
        datetime_format = '%Y\n%m-%d'
    else:
        datetime_format = '%Y-%m'
    return datetime_format

def set_datetime_ticker(dt):
    datetime_format = set_datetime_format(dt)
    tickers = dt.strftime(datetime_format)
    return tickers

def zoom_yaxis(ax, f):
    '''
    f : zoom factor
    f = 0 : unchange
    f < 0 : zoom in
    f > 0 : zoom out
    '''
    l = ax.get_ylim()
    new_l = (l[0] + l[1])/2 + np.array((-0.5, 0.5)) * (l[1] - l[0]) * (1 + f)
    ax.set_ylim(new_l)
    return ax

def plot_y(ax, y, label=None, linewidth=1):
    xax = np.arange(len(y))
    ax.plot(xax, y, label=label, linewidth=linewidth)
    ax.legend(loc='upper left')
    
def plot_bar(ax, h, width=0.2): #h: height of bar
    xax = np.arange(len(h))
    ax.bar(xax, h, width=width)
  
def plot_stack(ax, h, label=None): #h: weights of each asset
    xax = np.arange(len(h))
    ax.stackplot(xax, h.values.T, labels=label)
    ax.legend(loc='upper left')