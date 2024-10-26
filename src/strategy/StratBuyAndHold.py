from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from typing import Dict, List
import pandas as pd
import numpy as np

class StratBuyAndHold(IStrategy):
    def __init__(self, 
                 targetTickers: List[str]):
        self.targetTickers = targetTickers
        self.__assets: Dict[str, AssetData] = {}
        self.__portfolio: Portfolio = None

        self.__assetdateIdx: Dict[str, int] = {}

    def apply(self,
              assetdateIdx: Dict[str, int] = {}):
        self.__assetdateIdx = assetdateIdx

        for targetTicker in self.targetTickers:
            if self.__portfolio.positions.keys().__contains__(targetTicker):
                continue
                
            cashPerStock = np.floor(self.__portfolio.cash / len(self.targetTickers))
            asset = self.__assets.get(targetTicker)
            if asset is not None:
                priceData = asset.shareprice[self.__assetdateIdx[targetTicker]]
                if not priceData.empty:
                    price: float = float(priceData['Adj Close'])
                    quantity = np.floor((cashPerStock - ActionCost().buy(cashPerStock)) / price)
                    self.__portfolio.buy(targetTicker, quantity, price)
