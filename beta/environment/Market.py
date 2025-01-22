class Market():

    """
    Creates a market object that holds the market data for the simulation.
    """
    def __init__(self, data, n_assets, current_frame):
        """
        Inputs:
        - (pd.Dataframe) data: market data, it has to contain the column "sequential_date"
        - (int) n_assets: number of assets
        - (int) current_frame: current window frame
        """
        assert "sequential_date" in data.columns, "The data must contain the column 'sequential_date'"
        self.data = data
        self.n_assets = n_assets
        self.current_frame = current_frame

    def get_data(self, window_frame):
        """
        Get the data for the current window frame.
        """
        return self.data.loc[(self.data['date_sequential'] > window_frame) &
                         (self.data['date_sequential'] <= self.current_date)]
    
    def evolve(self):
        """
        Evolve the market to the next frame.
        """
        self.current_frame += 1