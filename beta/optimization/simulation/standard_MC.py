import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
importlib.reload(estimators)
from beta.forecaster.models import estimators

class PortfolioOptimizer():

    def __init__(self, returns, criterion = "sharpe_ratio", sampling="uniform", num_portfolios=5000, risk_free_rate=0.02):
        assert sampling in ["uniform", "normal"], "Sampling must be in ['uniform', 'normal']"
        self.returns = returns
        self.num_portfolios = num_portfolios
        self.risk_free_rate = risk_free_rate
        self.assets = returns.columns
        self.num_assets = len(self.assets)+1
        self.portfolio_weights = []
        self.portfolio_expected_returns = []
        self.portfolio_volatilities = []
        self.portfolio_performances = []
        self.sampling = sampling
        self.criterion = estimators.get_criterion(criterion)


    def generate_random_portfolios(self,  transaction_cost = 0.02, initial_guess = None, scale = 0.6):
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
            expected_return = np.concatenate([[self.risk_free_rate], estimators.sample_mean(self.returns)]) @ weights
            self.portfolio_expected_returns.append(expected_return)
            volatility = np.dot(weights[1:], np.dot(estimators.sample_covariance(self.returns), weights[1:]))*(1-weights[0])

            self.portfolio_volatilities.append(volatility)
            performance = self.criterion(self.returns, weights, self.risk_free_rate)
            assert performance.shape == (), f"Criterion function must return a scalar {performance.shape}"
            self.portfolio_performances.append(performance)
    def get_portfolio_performance(self, weights):
        weights = np.array(weights)
        expected_return = np.concatenate([[self.risk_free_rate], estimators.sample_mean(self.returns)]) @ weights
        volatility = np.dot(weights[1:], np.dot(estimators.sample_covariance(self.returns), weights[1:]))*(1-weights[0])
        performance = self.criterion(self.returns, weights, self.risk_free_rate)
        return expected_return, volatility, performance

    def plot_efficient_frontier(self):
        portfolios = pd.DataFrame({
            'Return': self.portfolio_expected_returns,
            'Volatility': self.portfolio_volatilities,
            'Criterion': self.portfolio_performances
        })

        # Plot efficient frontier
        plt.figure(figsize=(10, 7))
        plt.scatter(portfolios['Volatility'], portfolios['Return'], c=portfolios['Sharpe Ratio'], cmap='viridis')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')


        return portfolios
    
    def optimal_weights(self):
        assert len(self.portfolio_performances) > 0, "No portfolios generated"
        print(np.array(self.portfolio_performances).shape)
        print(len(self.portfolio_weights))
        index = np.argmax(self.portfolio_performances)
        print(index)
        return self.portfolio_weights[index]