from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
from typing import Dict
import pandas as pd
import numpy as np
import datetime

class StratBuyAndHold(IStrategy):
    def __init__(self, target_ticker: str):
        self.target_ticker = target_ticker.lower()
        self.has_bought = False

    def apply(self, assets: Dict[str, AssetData], portfolio: Portfolio, current_time: pd.Timestamp):
        assets = {key.lower(): value for key, value in assets.items()} #enforce lower case

        if portfolio.positions.keys().__contains__(self.target_ticker):
            return
        
        asset = assets.get(self.target_ticker)
        if asset is not None:
            price_data = asset.shareprice.loc[(asset.shareprice.index > current_time-pd.Timedelta(days=1)) & (asset.shareprice.index < current_time+pd.Timedelta(days=1))]
            if not price_data.empty:
                price: float = (float)(price_data['Close'])
                quantity = np.floor(portfolio.cash / price)
                portfolio.buy(self.target_ticker, quantity, price)
