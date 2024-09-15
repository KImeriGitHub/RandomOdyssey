from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
from typing import Dict
import datetime

class StratBuyAndHold(IStrategy):
    def __init__(self, target_ticker: str):
        self.target_ticker = target_ticker
        self.has_bought = False

    def apply(self, assets: Dict[str, AssetData], portfolio: Portfolio, current_time: datetime.datetime):
        if self.has_bought:
            return
        asset = assets.get(self.target_ticker)
        if asset is not None:
            price_data = asset.shareprice.loc[asset.shareprice.index == current_time]
            if not price_data.empty:
                price = price_data['Close'].values[0]
                quantity = portfolio.cash / price
                portfolio.buy(self.target_ticker, quantity, price)
                self.has_bought = True
