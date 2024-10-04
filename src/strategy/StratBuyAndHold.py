from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
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

        self.__stoplossLimit: Dict[str, float] = {}
        self.__blacklist: Dict[str, pd.Timestamp] = {}

        self.__currentDate: pd.Timestamp = pd.Timestamp(None)
        self.__assetdateIdx: Dict[str, int] = {}

    def apply(self,
              assets: Dict[str, AssetData], 
              portfolio: Portfolio, 
              currentDate: pd.Timestamp, 
              assetdateIdx: Dict[str, int] = {}):
        
        self.__assets = assets
        self.__portfolio = portfolio
        self.__currentDate = currentDate
        self.__assetdateIdx = assetdateIdx

        for targetTicker in self.targetTickers:
            if self.__portfolio.positions.keys().__contains__(targetTicker):
                continue
                
            cashPerStock = np.floor(self.__portfolio.cash / len(self.targetTickers))
            asset: AssetData = self.__assets.get(targetTicker)
            if asset is not None:
                priceData = asset.shareprice[self.__assetdateIdx[targetTicker]]
                if not priceData.empty:
                    price: float = float(priceData['Close'])
                    quantity = np.floor((cashPerStock - ActionCost().buy(cashPerStock)) / price)
                    self.__portfolio.buy(targetTicker, quantity, price)
