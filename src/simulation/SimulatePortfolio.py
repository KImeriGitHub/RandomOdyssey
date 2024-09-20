from typing import List
from src.common.Portfolio import Portfolio
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.simulation.ISimulation import ISimulation
import pandas as pd
import numpy as np

class SimulatePortfolio(ISimulation):
    def __init__(self, initialCash: float, strategy: IStrategy, assets: List[AssetData], startDate: pd.Timestamp, endDate: pd.Timestamp):
        self.portfolio = Portfolio(cash=initialCash)
        self.strategy = strategy
        self.assets = {asset.ticker: asset for asset in assets}

        if startDate.tzinfo is None:
            startDate = startDate.tz_localize('UTC')
        if endDate.tzinfo is None:
            endDate = endDate.tz_localize('UTC')

        self.startDate = startDate
        self.endDate = endDate

    def run(self):
        assetdateIdx = {}
        for ticker, asset in self.assets.items():
            if asset.shareprice.index[-1] < self.startDate:
                raise ValueError("Asset history not old enough or startDate too far back.")
            assetdateIdx[ticker] = np.argmax(asset.shareprice.index > self.startDate)

        dates = pd.date_range(self.startDate, self.endDate, freq='B')  # 'B' for business days
        for date in dates:
            date = date
            asset_prices = {}
            for ticker, asset in self.assets.items():
                price_data = pd.DataFrame(None)
                assetDate: pd.Timestamp = asset.shareprice.index[assetdateIdx[ticker]]
                if (assetDate > date - pd.Timedelta(days=1)) & (assetDate < date + pd.Timedelta(days=1)):
                    price_data = asset.shareprice.iloc[assetdateIdx[ticker]]
                    assetdateIdx[ticker] += 1
                if not price_data.empty:
                    price = price_data['Close']
                    asset_prices[ticker] = price
            # Apply the strategy
            self.strategy.apply(self.assets, self.portfolio, date)
            # Update portfolio value
            self.portfolio.update_value(date, asset_prices)

            if date> pd.Timestamp("2019-02-01", tz="UTC"):
                pass