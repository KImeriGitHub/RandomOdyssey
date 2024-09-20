# src/simulation/TopThreeSP500Strategy.py

from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
from typing import Dict
import datetime
import pandas as pd

class TopThreeSP500Strategy(IStrategy):
    def __init__(self):
        self.has_bought = False
        self.sp500_tickers = self._load_sp500_tickers()

    def _load_sp500_tickers(self) -> pd.DataFrame:
        """
        Loads the list of S&P 500 companies with their market capitalization.
        """
        # For the purpose of this example, let's assume we have a CSV file with S&P 500 tickers and market caps
        # In practice, you would retrieve this data from a reliable source or an API
        sp500_df = pd.read_csv('data/sp500_market_cap.csv')  # Columns: ['Ticker', 'MarketCap']
        sp500_df.sort_values(by='MarketCap', ascending=False, inplace=True)
        return sp500_df

    def apply(self, assets: Dict[str, AssetData], portfolio: Portfolio, current_time: datetime.datetime):
        if self.has_bought:
            return

        # Get the top three tickers by market cap
        top_tickers = self.sp500_tickers['Ticker'].head(3).tolist()

        # Check if the assets are available
        available_tickers = [ticker for ticker in top_tickers if ticker in assets]

        if not available_tickers:
            print("No top S&P 500 tickers available in assets.")
            return

        # Divide cash equally among available tickers
        cash_per_stock = portfolio.cash / len(available_tickers)

        for ticker in available_tickers:
            asset = assets[ticker]
            price_data = asset.shareprice.loc[asset.shareprice.index == current_time]
            if not price_data.empty:
                price = price_data['Close'].values[0]
                quantity = cash_per_stock / price
                portfolio.buy(ticker, quantity, price)
                print(f"Bought {quantity} shares of {ticker} at {price}")
            else:
                print(f"No price data for {ticker} on {current_time}")

        self.has_bought = True
