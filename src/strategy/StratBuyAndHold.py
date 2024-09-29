from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from typing import Dict, List
import pandas as pd
import numpy as np
import datetime

class StratBuyAndHold(IStrategy):
    def __init__(self, targetTickers: List[str]):
        self.targetTickers = [ticker.lower() for ticker in targetTickers]

    def apply(self, assets: Dict[str, AssetData], portfolio: Portfolio, current_time: pd.Timestamp, assetdateIdx = None):
        assets = {key.lower(): value for key, value in assets.items()} #enforce lower case

        for targetTicker in self.targetTickers:
            if portfolio.positions.keys().__contains__(targetTicker):
                continue
                
            cash_per_stock = np.floor(portfolio.cash / len(self.targetTickers))
            asset = assets.get(targetTicker)
            if asset is not None:
                timeintervalIdx = (asset.shareprice.index >= current_time-pd.Timedelta(hours=18)) & \
                                  (asset.shareprice.index < current_time+pd.Timedelta(hours=18))
                price_data = asset.shareprice.loc[timeintervalIdx]
                if not price_data.empty:
                    price: float = float(price_data['Close'].iloc[0])
                    quantity = np.floor((cash_per_stock - ActionCost().buy(cash_per_stock)) / price)
                    portfolio.buy(targetTicker, quantity, price)
