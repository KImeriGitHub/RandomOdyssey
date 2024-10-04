import matplotlib.pyplot as plt
import pandas as pd
from src.common.Portfolio import Portfolio

class ResultAnalyzer:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def plot_portfolio_value(self):
        df = pd.DataFrame(self.portfolio.valueOverTime, columns=["Timestamp", "Value"]).set_index("Timestamp")
        df.plot()
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()
