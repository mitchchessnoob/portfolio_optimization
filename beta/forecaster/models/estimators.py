import numpy as np
import pandas as pd
import datetime as dt

def get_criterion(criterion):
    """
    Get the criterion function.
    Inputs:
    - (str) criterion: the criterion function

    Returns: 
     - (function) the criterion function
    """
    implemented = {
        "sharpe_ratio": sharpe_ratio,
        "sample_mean": sample_mean,
        "sample_covariance": sample_covariance,
        "sortino_ratio": sortino_ratio
    }

    # Check if the criterion is implemented
    if criterion not in implemented:
        raise ValueError(f"Criterion '{criterion}' is not implemented. Available: {list(implemented.keys())}")

    return implemented[criterion]

def sample_mean(data):
    """
    Compute the sample mean of the data.
    Inputs:
    - (np.array) data: the data in format (n_samples, n_assets)

    Returns: 
     - (np.array) the sample mean of every asset
    """
    return np.mean(data, axis=0)

def sample_covariance(data):
    """
    Compute the sample covariance of the data.
    Inputs:
    - (np.array) data: the data in format (n_samples, n_assets)

    Returns: 
     - (np.array) the sample covariance matrix
    """
    return np.cov(data.T)

def sharpe_ratio(data, weights, rf = 0.02):
    """
    Compute the sharpe ratio of the data.
    Inputs:
    - (np.array) data: the data in format (n_samples, n_assets)
    - (np.array) weights: the weights of the portfolio of shape (n_assets)
    - (float) rf: the risk free rate

    Returns: 
     - (float) the sharpe ratio
    """
    return (np.dot(np.mean(data),weights)*255 - rf)/(np.sqrt(255*np.dot(np.dot(weights, sample_covariance(data)), weights)))


def sortino_ratio(data, weights, rf = 0.02):
    """
    Compute the sharpe ratio of the data.
    Inputs:
    - (np.array) data: the data in format (n_samples, n_assets)
    - (np.array) weights: the weights of the portfolio of shape (n_assets)
    - (float) rf: the risk free rate

    Returns: 
     - (float) the sharpe ratio
    """
    return (np.dot(np.mean(data),weights)*255 - rf)/(np.sqrt(255*np.dot(np.dot(weights, sample_covariance(data[data<0])), weights)))