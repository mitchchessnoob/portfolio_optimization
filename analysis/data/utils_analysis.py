import numpy as np

import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

def download_data_yf(tickers, start_date, end_date):
    """
    Download the data from Yahoo Finance.
    Inputs:
    - ([str]) tickers: list of tickers of the assets
    - (str) start_date: start date of the data
    - (str) end_date: end date of the data
    """
    
    data = yf.download(tickers=tickers, start=start_date, end=end_date, interval="1d", group_by='Ticker', auto_adjust=True, prepost=True, threads=True, proxy=None)
    data.dropna(inplace=True)

    # Reshape the DataFrame
    data = data.stack(level=0).reset_index()
    data.rename(columns={'level_1': 'Ticker'}, inplace=True)
    return data

def create_full_dataset(data, ESG_dataset):
    """
    Create the full dataset.
    Inputs:
    - (pd.DataFrame) data: market data
    - (pd.DataFrame) ESG_values: ESG values
    """
    # Merge the data
    ESG_dataset.rename(columns={'ticker': 'Ticker'}, inplace=True)
    ESG_dataset['Ticker'] = ESG_dataset['Ticker'].str.upper()

    
    return data.join(ESG_dataset.set_index('Ticker'), on='Ticker')

def get_aggregate_stats(data):
    data

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