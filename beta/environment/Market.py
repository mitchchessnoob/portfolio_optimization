import pandas as pd
import datetime as dt


class Market():

    """
    Creates a market object that holds the market data for the simulation.
    """
    def __init__(self, data, n_assets, current_day):
        """
        Inputs:
        - (pd.Dataframe) data: market data, has to contain the column 'Date'
        - (int) n_assets: number of assets
        - (date) current_frame: current window frame
        """
        assert "Date" in data.columns, "The data must contain the column 'Date'"
        self.data = data
        self.n_assets = n_assets
        self.current_day = current_day

    def get_data(self, window_frame):
        """
        Get the data from the present day t to day t-window_frame included

        Returns: data regarding the last window_frame + 1 days
        """
        return self.data.loc[(self.data['Date'] >= self.current_date - dt.timedelta(days=window_frame)) &
                         (self.data['date_sequential'] <= self.current_date)]
    
    def evolve(self):
        """
        Evolve the market to the next frame.
        """
        self.current_frame += 1
    
    def get_today_returns(self):
        """
        Get the returns for the current frame.
        """
        return self.data.loc[self.data['date_sequential'] == self.current_frame]

    def influence_portfolio(self, portfolio):
        """
        Influence the portfolio with the market data.
        Inputs:
        - (Portfolio) portfolio: portfolio object
        """
        pass
        
