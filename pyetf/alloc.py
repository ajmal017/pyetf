# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas.core.base import PandasObject
from scipy.optimize import minimize
from decorator import decorator
from sklearn.covariance import ledoit_wolf

@decorator
def mean_var_weights(func_covar, *args, **kwargs):
    """
    Calculates the mean-variance weights given a DataFrame of returns.

    Args:
        * args[0]: returns (DataFrame): Returns for multiple securities.
        * args[1]: weight_bounds ((low, high)): Weigh limits for optimization.
        * args[2]: rf (float): `Risk-free rate <https://www.investopedia.com/terms/r/risk-freerate.asp>`_ used in utility calculation
        * args[3]: options (dict): options for minimizing, e.g. {'maxiter': 10000 }

    Returns:
        Series {col_name: weight}
    """
    if len(args)<4:
        raise Exception("Not Enough Parameters")
    returns = args[0]
    weight_bounds = args[1]
    rf = args[2]
    options = args[3]
    
    def fitness(weights, exp_rets, covar, rf):
        # portfolio mean
        mean = sum(exp_rets * weights)
        # portfolio var
        var = np.dot(np.dot(weights, covar), weights)
        # utility - i.e. sharpe ratio
        util = (mean - rf) / np.sqrt(var)
        # negative because we want to maximize and optimizer
        # minimizes metric
        return -util

    n = len(returns.columns)

    # expected return defaults to mean return by default
    exp_rets = returns.mean()

    # calc covariance matrix       
    covar = func_covar(returns)

    weights = np.ones([n]) / n
    bounds = [weight_bounds for i in range(n)]
    # sum of weights must be equal to 1
    constraints = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
    optimized = minimize(fitness, weights, (exp_rets, covar, rf),
                         method='SLSQP', constraints=constraints,
                         bounds=bounds, options=options)
    # check if success
    if not optimized.success:
        raise Exception(optimized.message)

    # return weight vector
    return pd.Series({returns.columns[i]: optimized.x[i] for i in range(n)})

@mean_var_weights
def mvw_standard(prices, 
                 weight_bounds=(0.,1.),
                 rf = 0.,
                 options = None):
    """
    Calculates the mean-variance weights given a DataFrame of returns.
    Wraps mean_var_weights with standard covariance calculation method

    Args:
        * prices (DataFrame): Prices for multiple securities.
        * weight_bounds ((low, high)): Weigh limits for optimization.
        * rf (float): `Risk-free rate <https://www.investopedia.com/terms/r/risk-freerate.asp>`_ used in utility calculation
        * options (dict): options for minimizing, e.g. {'maxiter': 10000 }

    Returns:
        Series {col_name: weight}
    """
    r = prices.to_returns().dropna()    
    covar = r.cov()
    return covar

@mean_var_weights
def mvw_ledoit_wolf(prices, 
                 weight_bounds=(0.,1.),
                 rf = 0.,
                 options = None):
    """
    Calculates the mean-variance weights given a DataFrame of returns.
    Wraps mean_var_weights with ledoit_wolf covariance calculation method

    Args:
        * prices (DataFrame): Prices for multiple securities.
        * weight_bounds ((low, high)): Weigh limits for optimization.
        * rf (float): `Risk-free rate <https://www.investopedia.com/terms/r/risk-freerate.asp>`_ used in utility calculation
        * options (dict): options for minimizing, e.g. {'maxiter': 10000 }

    Returns:
        Series {col_name: weight}
    """
    r = prices.to_returns().dropna()
    covar = ledoit_wolf(r)[0]
    return covar

def _erc_weights_ccd(x0,
                     cov,
                     b,
                     maximum_iterations,
                     tolerance):
    """
    Calculates the equal risk contribution / risk parity weights given
    a DataFrame of returns.

    Args:
        * x0 (np.array): Starting asset weights.
        * cov (np.array): covariance matrix.
        * b (np.array): Risk target weights.
        * maximum_iterations (int): Maximum iterations in iterative solutions.
        * tolerance (float): Tolerance level in iterative solutions.

    Returns:
        np.array {weight}

    Reference:
        Griveau-Billion, Theophile and Richard, Jean-Charles and Roncalli,
        Thierry, A Fast Algorithm for Computing High-Dimensional Risk Parity
        Portfolios (2013).
        Available at SSRN: https://ssrn.com/abstract=2325255

    """
    n = len(x0)
    x = x0.copy()
    var = np.diagonal(cov)
    ctr = cov.dot(x)
    sigma_x = np.sqrt(x.T.dot(ctr))

    for iteration in range(maximum_iterations):

        for i in range(n):
            alpha = var[i]
            beta = ctr[i] - x[i] * alpha
            gamma = -b[i] * sigma_x

            x_tilde = (-beta + np.sqrt(
                beta * beta - 4 * alpha * gamma)) / (2 * alpha)
            x_i = x[i]

            ctr = ctr - cov[i] * x_i + cov[i] * x_tilde
            sigma_x = sigma_x * sigma_x - 2 * x_i * cov[i].dot(
                x) + x_i * x_i * var[i]
            x[i] = x_tilde
            sigma_x = np.sqrt(sigma_x + 2 * x_tilde * cov[i].dot(
                x) - x_tilde * x_tilde * var[i])

        # check convergence
        if np.power((x - x0) / x.sum(), 2).sum() < tolerance:
            return x / x.sum()

        x0 = x.copy()

    # no solution found
    raise ValueError('No solution found after {0} iterations.'.format(
        maximum_iterations))

@decorator
def risk_parity_weights(func_covar, *args, **kwargs):                     
    """
    Calculates the equal risk contribution / risk parity weights given a
    DataFrame of returns.

    Args:
        * args[0]: returns (DataFrame): Returns for multiple securities.
        * args[1]: initial_weights (list): Starting asset weights [default inverse vol].
        * args[2]: risk_weights (list): Risk target weights [default equal weight].
        * args[3]: risk_parity_method (str): Risk parity estimation method.
            Currently supported:
                - ccd (cyclical coordinate descent)[default]
        * args[4]: maximum_iterations (int): Maximum iterations in iterative solutions.
        * args[5]: tolerance (float): Tolerance level in iterative solutions.

    Returns:
        Series {col_name: weight}

    """
    if len(args)<6:
        raise Exception("Not Enough Parameters")
    returns = args[0]
    initial_weights = args[1]
    risk_weights = args[2]
    risk_parity_method = args[3]
    maximum_iterations = args[4]
    tolerance = args[5]
    
    n = len(returns.columns)

    # calc covariance matrix
    covar = func_covar(returns)

    # initial weights (default to inverse vol)
    if initial_weights is None:
        inv_vol = 1. / np.sqrt(np.diagonal(covar))
        initial_weights = inv_vol / inv_vol.sum()

    # default to equal risk weight
    if risk_weights is None:
        risk_weights = np.ones(n) / n

    # calc risk parity weights matrix
    if risk_parity_method == 'ccd':
        # cyclical coordinate descent implementation
        erc_weights = _erc_weights_ccd(
            initial_weights,
            covar,
            risk_weights,
            maximum_iterations,
            tolerance
        )    
    else:
        raise NotImplementedError('risk_parity_method not implemented')

    # return erc weights vector
    return pd.Series(erc_weights, index=returns.columns, name='erc')

@risk_parity_weights
def rpw_standard(prices,
                 initial_weights = None,
                 risk_weights = None,
                 risk_parity_method = 'ccd',
                 maximum_iterations = 100,
                 tolerance = 1E-8):
    """
    Calculates the equal risk contribution / risk parity weights given a
    DataFrame of returns.
    Wraps mean_var_weights with standard covariance calculation method

    Args:
        * prices (DataFrame): Prices for multiple securities.
        * initial_weights (list): Starting asset weights [default inverse vol].
        * risk_weights (list): Risk target weights [default equal weight].
        * risk_parity_method (str): Risk parity estimation method.
            Currently supported:
                - ccd (cyclical coordinate descent)[default]
        * maximum_iterations (int): Maximum iterations in iterative solutions.
        * tolerance (float): Tolerance level in iterative solutions.

    Returns:
        Series {col_name: weight}

    """
    r = prices.to_returns().dropna()
    covar = r.cov().values
    return covar

@risk_parity_weights
def rpw_ledoit_wolf(prices,
                    initial_weights = None,
                    risk_weights = None,
                    risk_parity_method = 'ccd',
                    maximum_iterations = 100,
                    tolerance = 1E-8):
    """
    Calculates the equal risk contribution / risk parity weights given a
    DataFrame of returns.
    Wraps mean_var_weights with ledoit_wolf covariance calculation method

    Args:
        * prices (DataFrame): Prices for multiple securities.
        * initial_weights (list): Starting asset weights [default inverse vol].
        * risk_weights (list): Risk target weights [default equal weight].
        * risk_parity_method (str): Risk parity estimation method.
            Currently supported:
                - ccd (cyclical coordinate descent)[default]
        * maximum_iterations (int): Maximum iterations in iterative solutions.
        * tolerance (float): Tolerance level in iterative solutions.

    Returns:
        Series {col_name: weight}

    """
    r = prices.to_returns().dropna()
    covar = ledoit_wolf(r)[0]
    return covar

def to_weights(
        prices, 
        func_weighting=rpw_standard, 
        hist_length=200, 
        *args, **kwargs):
    """    
    Calculates the weights of each asset in portfolio.
    Use historical data since 0:hist_length-1 to calculate weights at hist_length
    In Python, data in [0:hist_length] -> weights stored at [hist_length-1]
    
    Ex: 
        prices.to_weights()
        prices.to_weights(func_weighting=rpw_ledoit_wolf, risk_weights=mc)
        
    Args:
        * prices (DataFrame): Prices for multiple securities.
        * func_weighting (function): function to use to estimate covariance
            [default rpw_standard].
        * hist_length: Length of data to use [default 200].
        * other paramters: Used in func_weighting.
            i.e. mc=[0.5, 0.3, 0.2]
            risk_weights=mc

    Returns:
        Pandas Dataframe
    """
    w = prices.copy()
    m = hist_length
    
    for e in w.columns:
        w[e] = np.nan
    
    # 0:m -> m
    # python - [0:m] -> [m-1]  
    for t in range(0, len(prices)-m+1):
        p = prices.iloc[t:t+m]    
        w.iloc[t+m-1] = func_weighting(p, *args, **kwargs)
    
    return w

def to_NAV(prices, weights, init_cash=1000000):
    portfolio = prices.copy()
    w = weights.copy()
    portfolio['NAV'] = w.sum(axis=1)
    portfolio = portfolio[portfolio.NAV>0]
    w = w.dropna()
    portfolio.iloc[0].NAV = init_cash
    s = w.copy()        
    for e in w.columns:
        s.iloc[0][e] = 0.
    for t in range(1, len(w)):
        nav = portfolio.iloc[t-1].NAV
        for e in w.columns:
            s.iloc[t][e] = nav * w.iloc[t-1][e] / portfolio.iloc[t-1][e]
        nav = 0.
        for e in w.columns:
            nav += s.iloc[t][e] * portfolio.iloc[t][e]
        portfolio.iloc[t].NAV = nav
    return portfolio

def to_NAV2(prices, weights, init_cash=1000000, commission=0.01):
    portfolio = prices.copy()
    w = weights.copy()
    portfolio['NAV'] = w.sum(axis=1)
    portfolio = portfolio[portfolio.NAV>0]
    w = w.dropna()
    portfolio.iloc[0].NAV = init_cash    
    s = to_shares(prices, weights)
    cash = init_cash
    for t in range(1, len(w)):
        nav = 0.
        for e in w.columns:
            nav += s.iloc[t][e] * portfolio.iloc[t][e]
            delta_cash = -(s.iloc[t][e]-s.iloc[t-1][e]) * portfolio.iloc[t][e]
            cash += delta_cash - abs(delta_cash)*commission
        portfolio.iloc[t].NAV = nav + cash
    return portfolio

def to_shares(prices, weights, init_cash=1000000, nShares=100):
    portfolio = prices.copy()
    w = weights.copy()
    portfolio['NAV'] = w.sum(axis=1)
    portfolio = portfolio[portfolio.NAV>0]
    w = w.dropna()
    s = w.copy()
    portfolio.iloc[0].NAV = init_cash
    for e in w.columns:
        s.iloc[0][e] = 0.
    for t in range(1, len(w)):
        nav = portfolio.iloc[t-1].NAV
        for e in w.columns:
            s.iloc[t][e] = nav * w.iloc[t-1][e] / portfolio.iloc[t-1][e]
        nav = 0.
        for e in w.columns:
            nav += s.iloc[t][e] * portfolio.iloc[t][e]
        portfolio.iloc[t].NAV = nav
    s = s // nShares * nShares
    return s

def to_shares2(prices, weights, init_cash=1000000, nShares=100):
    portfolio = prices.copy()
    w = weights.copy()
    portfolio['NAV'] = w.sum(axis=1)
    portfolio = portfolio[portfolio.NAV>0]
    w = w.dropna()
    s = w.copy()
    portfolio.iloc[0].NAV = init_cash
    for e in w.columns:
        s.iloc[0][e] = 0.
    for t in range(1, len(w)):
        nav = portfolio.iloc[t-1].NAV
        for e in w.columns:
            s.iloc[t][e] = nav * w.iloc[t-1][e] / portfolio.iloc[t-1][e]
        nav = 0.
        for e in w.columns:
            nav += s.iloc[t][e] * portfolio.iloc[t][e]
            s.iloc[t][e] = (s.iloc[t][e]//nShares)*nShares
        portfolio.iloc[t].NAV = nav
    return s

def extend_pandas():
    """
    Extends pandas function, i.e. display items
    Ex:
        prices.to_weights()
        (where prices would be a DataFrame)
    """
    PandasObject.to_weights = to_weights
    PandasObject.to_shares = to_shares
    PandasObject.to_NAV = to_NAV
    PandasObject.to_NAV2 = to_NAV2