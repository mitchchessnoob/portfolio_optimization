import numpy as np


class Portfolio():
    
    def __init__(self, weights, current_value, risk_free_rate = 0.03,  transaction_cost = 0.02):
        """
        Creates a portfolio object that holds the portfolio data for the simulation, the first entry is the weight in bond portfolio, if
         risk_free_rate = 0 then the position is in cash.
        """
        assert np.sum(weights) == 1, "Sum of the weights must be 1"
        assert np.sum(weights < 0 ) == 0, "Only positive weights are allowed"
        self.weights = np.array(weights)
        self.current_value = current_value
        self.risk_free_rate = (1+risk_free_rate)**(1/364) - 1
        self.transaction_cost = transaction_cost
        self.last_return = 0
    
    # def update_weights(self, new_weights):
    #     """
    #     Update the weights of the portfolio.
    #     """
    #     self.weights = new_weights
    
    def reallocate_assets(self, new_weights):
        """
        Reallocate the assets in the portfolio.
        """
        # assert np.sum(new_weights)*current_value + self.cash + self.bonds_value == new_value, "New position value not equal to the new capital"
        if np.sum(new_weights) != 1:
            print("Given weights are invalid, keeping previous weights")
            return self.weights
        if np.sum(new_weights < 0 ) != 0:
            print("Only positive weights are allowed, keeping previous weights")
            return self.weights
        new_weights = np.array(new_weights)
        self.current_value = self.current_value*(1 - self.transaction_cost*np.sum(np.abs(new_weights - self.weights)))
        self.weights = new_weights
    
    def market_changes(self, returns):
        """
        Update the portfolio value based on the market changes.
        Input: 
        - (pd.Dataframe) returns: a dataframe with the returns of the assets
        """
        returns = returns[sorted(returns.columns)]
        returns = np.array(returns).squeeze()
        returns = np.append(self.risk_free_rate, returns)
        self.weights = self.weights*(1 + returns)
        self.last_return = float(np.sum(self.weights))
        self.current_value = np.sum(self.weights)*self.current_value
        self.weights = self.weights/np.sum(self.weights)
        

        