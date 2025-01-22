import numpy as np


class Portfolio():
    def __init__(self, weights, cash, bonds_value, current_value,  transaction_cost = 0.02):
        """
        Creates a portfolio object that holds the portfolio data for the simulation.
        """
        assert np.sum(weights)*current_value + cash == current_value, "Initial position value not equal to the initial capital"
        self.weights = weights
        self.cash = cash
        self.current_value = current_value
        self.transaction_cost = transaction_cost
    
    def update_weights(self, new_weights):
        """
        Update the weights of the portfolio.
        """
        self.weights = new_weights
    
    def reallocate_assets(self, new_weights):
        """
        Reallocate the assets in the portfolio.
        """
        # assert np.sum(new_weights)*current_value + self.cash + self.bonds_value == new_value, "New position value not equal to the new capital"
        self.current_value = self.current_value*(1- self.transaction_cost*np.sum(np.abs(new_weights - self.weights)))
        self.cash = (1 - np.sum(new_weights))*self.current_value
        self.weights = new_weights
        