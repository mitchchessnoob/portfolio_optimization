import numpy as np
import pandas as pd
import yfinance as yf
import os
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def download_data_yf(tickers, start_date, end_date):
    """
    Download the data from Yahoo Finance.
    Inputs:
    - ([str]) tickers: list of tickers of the assets
    - (str) start_date: start date of the data
    - (str) end_date: end date of the data
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    data.dropna(inplace=True)

    # Reshape the DataFrame
    data = data.stack(level=0).reset_index()
    data.rename(columns={'level_1': 'Ticker'}, inplace=True)
    return data

def scale_data(data):
    """
    Scale the data and returns a pd.Dataframe with the same structure as the input one
    """
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    # Dictionary to store scalers for each ticker
    scalers = {}

    # Define a function to scale each group and store the scaler
    def scale_group(group):
        scaler = MinMaxScaler(feature_range=(0, 1))  # Create a new scaler for each group
        scalers[group.name] = scaler  # Store the scaler using the group name (ticker)
        group[numeric_cols] = scaler.fit_transform(group[numeric_cols])
        return group

    # Apply scaling to each group
    data = data.groupby('Ticker', group_keys=False).apply(scale_group)
    return data, scalers

def create_dataset_lstm(data, window_size, features):
    """
    Create the dataset for the LSTM model.
    Inputs:
    - (pd.DataFrame) data: market data
    - (int) window_size: size of the window
    - ([str]) features: list of features to use
    """
    x_train = []
    y_train = []
    for company in data["Ticker"].unique():
        data_company = data[data["Ticker"] == company]
        for i in range(window_size, len(data_company)):
            x_train.append(data_company[i-window_size:i][features])
            y_train.append(data_company[i:i+1]["Close"])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape x_train to a 3D array with the appropriate dimensions for the LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(features)))
    return x_train, y_train