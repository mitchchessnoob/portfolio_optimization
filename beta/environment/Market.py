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
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.days = sorted(self.data['Date'].unique())
        self.n_assets = n_assets
        if current_day not in self.days:
            print("The current day must be in the data, looking for the closest day")
            current_day = min(self.days, key=lambda x: abs(x - current_day))
            print(f"Current day set to {current_day}")
        self.current_day = current_day

    
    def today(self):
        """
        Get the current day.
        """
        return self.current_day.date()

    def get_data(self, window_frame):
        """
        Get the data from the present day t to day t-window_frame (both included).

        Returns: data regarding the last window_frame + 1 days of trading
        """
        index = self.days.index(self.current_day)
        first_day = self.days[index - window_frame]
        return self.data.loc[(self.data['Date'] >= first_day) &
                         (self.data['Date'] <= self.current_day)]
    
    def evolve(self):
        """
        Evolve the market to the next frame.
        """
        index = self.days.index(self.current_day)
        self.current_day = self.days[index + 1]
        return self.current_day

    def get_n_past_returns(self, n):
        """
        Get the n past returns for each asset.

        Returns: a pd.DataFrame with the columns 'Date', 'Ticker1', 'Ticker2', ..., 'TickerN'
        """
        data = self.get_data(n).copy()
        data = data.sort_values(by=['Ticker', 'Date'])

        # Group by Ticker and calculate percentage change
        data['Pct_Change'] = data.groupby('Ticker')['Close'].pct_change()

        # Pivot the DataFrame to get tickers as columns
        pivoted_data = data.pivot(index='Date', columns='Ticker', values='Pct_Change')

        # Optional: Rename columns to include "_pct_change" for clarity
        pivoted_data.columns = [f"{ticker}" for ticker in pivoted_data.columns]

        # Reset the index if needed
        pivoted_data.reset_index(inplace=True)
        return pivoted_data.dropna()
    
    def get_today_returns(self):
        """
        Get the returns for the current frame.
        Returns: a pd.DataFrame with the columns 'Ticker1', 'Ticker2', ..., 'TickerN'
        """
        data = self.get_data(1).copy()

        # Group by Ticker and calculate percentage change
        data['Pct_Change'] = data.groupby('Ticker')['Close'].pct_change()

        # Pivot the DataFrame to get tickers as columns
        pivoted_data = data.pivot(index='Date', columns='Ticker', values='Pct_Change')

        # Optional: Rename columns to include "_pct_change" for clarity
        pivoted_data.columns = [f"{ticker}" for ticker in pivoted_data.columns]

        # Reset the index if needed
        pivoted_data.reset_index(inplace=True)

        return pivoted_data.drop('Date', axis=1).dropna()

    def influence_portfolio(self, portfolio):
        """
        Influence the portfolio with the market data.
        Inputs:
        - (Portfolio) portfolio: portfolio object
        """
        returns = self.get_today_returns()
        portfolio.market_changes(returns)
        return portfolio
        
