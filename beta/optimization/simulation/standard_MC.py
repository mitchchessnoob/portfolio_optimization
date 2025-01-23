import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class PortfolioOptimizer():
    def __init__(self, returns, sampling="uniform", num_portfolios=5000, risk_free_rate=0.02):
        assert sampling in ["uniform", "normal"], "Sampling must be in ['uniform', 'normal']"
        self.returns = returns
        self.num_portfolios = num_portfolios
        self.risk_free_rate = risk_free_rate
        self.assets = returns.columns
        self.num_assets = len(self.assets)
        self.portfolio_weights = []
        self.portfolio_expected_returns = []
        self.portfolio_volatilities = []
        self.portfolio_sharpe_ratios = []
        self.sampling = sampling


    def generate_random_portfolios(self, transaction_cost = 0.02, initial_guess = None, scale = 0.6):
        if initial_guess is None:
            initial_guess = np.ones(self.num_assets) / self.num_assets
        np.random.seed(42)
        for _ in range(self.num_portfolios):
            if self.sampling == "uniform":
                weights = np.random.uniform(0, 1, self.num_assets)
                weights = weights / np.sum(weights)
            elif self.sampling == "normal":
                weights = np.random.normal(loc=initial_guess, scale =  scale*np.eye(self.num_assets))
                weights = np.max(weights, 0)
            weights /= np.sum(weights)
            weights /= np.sum(weights)
            self.portfolio_weights.append(weights)
            expected_return = np.sum(weights * self.returns.mean()) * 252
            self.portfolio_expected_returns.append(expected_return)
            volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
            self.portfolio_volatilities.append(volatility)
            sharpe_ratio = (expected_return - self.risk_free_rate - transaction_cost*np.sum(np.abs(initial_guess-weights)) ) / volatility
            self.portfolio_sharpe_ratios.append(sharpe_ratio)

    def get_portfolio_performance(self, weights):
        weights = np.array(weights)
        expected_return = np.sum(weights * self.returns.mean()) * 252
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        return expected_return, volatility, sharpe_ratio

    def plot_efficient_frontier(self):
        portfolios = pd.DataFrame({
            'Return': self.portfolio_expected_returns,
            'Volatility': self.portfolio_volatilities,
            'Sharpe Ratio': self.portfolio_sharpe_ratios
        })

        # Plot efficient frontier
        plt.figure(figsize=(10, 7))
        plt.scatter(portfolios['Volatility'], portfolios['Return'], c=portfolios['Sharpe Ratio'], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')


        return portfolios