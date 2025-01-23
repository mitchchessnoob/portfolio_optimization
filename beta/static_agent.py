import numpy as np
import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

from data.utils import download_data_yf
import importlib
import environment.Market
importlib.reload(environment.Market)
from environment.Market import Market
import environment.Portfolio
importlib.reload(environment.Portfolio)
from environment.Portfolio import Portfolio
from omegaconf import OmegaConf

def simulation(configs_path, initial_weights = None):
    """
    Simulate the environment.
    Inputs:
    - (yaml) configs: configuration file
    """
    
    with open(configs_path, "r") as f:
          yaml_data = f.read()
    configs = OmegaConf.create(yaml_data)
    
    if configs.data.yahoo_finance:
        tickers = list(configs.data.tickers)
        data = download_data_yf(tickers, configs.data.start_date, configs.data.end_date)
    else:
         data = pd.read_csv(configs.data.data_path)

    if configs.portfolio.initial_weights == "uniform":
        initial_weights = np.ones(len(tickers)+1)/(len(tickers)+1)
    elif configs.portfolio.initial_weights == "random":
        initial_weights = np.random.rand(len(tickers)+1)
        initial_weights = initial_weights/np.sum(initial_weights)
    elif configs.portfolio.initial_weights == "custom":
        assert initial_weights is not None, "Initial weights must be provided or be in {uniform, random}"
        initial_weights = np.array(initial_weights)
    day = configs.market.current_day.split("-")
    mkt = Market(data, len(tickers), dt.datetime(int(day[0]), int(day[1]), int(day[2])))
    ptf = Portfolio(initial_weights, configs.portfolio.starting_value, configs.portfolio.risk_free_rate, configs.portfolio.transaction_cost)

    metrics = {
        "days": [],
        "portfolio_value": [],
        "portfolio_weights": [],
        "portfolio_returns": [],
        "asset_returns": []
    }

    end_simulation_day = configs.simulation.end_simulation_day.split("-")
    end_simulation_day = dt.datetime(int(end_simulation_day[0]), int(end_simulation_day[1]), int(end_simulation_day[2]))
    for i in list(np.array(mkt.days)[np.array(mkt.days) < end_simulation_day ]):
        metrics["days"].append(mkt.today())
        mkt.evolve()
        ptf = mkt.influence_portfolio(ptf)
        metrics["portfolio_value"].append(ptf.current_value)
        metrics["portfolio_weights"].append(ptf.weights)
        metrics["portfolio_returns"].append(ptf.last_return)
        metrics["asset_returns"].append(np.array(mkt.get_today_returns()).squeeze())
    
    return metrics