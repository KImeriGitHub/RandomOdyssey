import matplotlib.pyplot as plt
from src.common.Portfolio import Portfolio

class ResultAnalyzer:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def plot_portfolio_value(self):
        self.portfolio.history.set_index('Date', inplace=True)
        self.portfolio.history['Value'].plot()
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()
