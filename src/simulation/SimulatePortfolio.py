from typing import List
from src.common.Portfolio import Portfolio
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.simulation.ISimulation import ISimulation
import pandas as pd
import datetime

class SimulatePortfolio(ISimulation):
    def __init__(self, initial_cash: float, strategy: IStrategy, assets: List[AssetData], start_date: datetime.datetime, end_date: datetime.datetime):
        self.portfolio = Portfolio(cash=initial_cash)
        self.strategy = strategy
        self.assets = {asset.ticker: asset for asset in assets}
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        dates = pd.date_range(self.start_date, self.end_date, freq='B')  # 'B' for business days
        for date in dates:
            # For each date, get prices of assets
            asset_prices = {}
            for ticker, asset in self.assets.items():
                # Get the price for the date
                try:
                    price_data = asset.shareprice.loc[asset.shareprice.index == date]
                    if not price_data.empty:
                        price = price_data['Close'].values[0]
                        asset_prices[ticker] = price
                except KeyError:
                    continue  # No data for this date
            # Apply the strategy
            self.strategy.apply(self.assets, self.portfolio, date)
            # Update portfolio value
            self.portfolio.update_value(date, asset_prices)
