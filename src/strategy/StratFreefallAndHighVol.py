from src.mathTools.CurveAnalysis import CurveAnalysis
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.predictionModule.FourierML import FourierML
from typing import Dict, List
import pandas as pd
import polars as pl
import numpy as np
from sklearn.utils import shuffle

class StratFreefallAndHighVol(IStrategy):
    __cashthreshhold = 0.2
    __stockTracker: Dict[str, dict] = {}

    def __init__(self,
                 num_choices: int = 1,
                 portfolio: Portfolio = None,
                 assets: Dict[str, AssetDataPolars] = {}):

        self.__num_choices: int = num_choices

        self.__assets: Dict[str, AssetDataPolars] = assets
        self.__portfolio: Portfolio = portfolio

        self.__assetdateIdx: Dict[str, int] = {}
        
        self.printBuySell = False

    def sellOrders(self) -> Dict:
        sellOrders = {}
        for boughtTicker in self.__portfolio.positions.keys():
            asset: AssetDataPolars = self.__assets[boughtTicker]
            aIdx = self.__assetdateIdx[boughtTicker]
            price = asset.adjClosePrice['AdjClose'].item(aIdx)
            price_m1 = asset.adjClosePrice['AdjClose'].item(aIdx-1)

            boughtPrice = self.__stockTracker[boughtTicker].get("boughtPrice", 0)
            boughtIdx = self.__stockTracker[boughtTicker].get("boughtDateIdx")
            if (aIdx - boughtIdx > 21 and price<price_m1):
                sellOrders[boughtTicker] = {
                    "quantity": self.__portfolio.positions[boughtTicker],
                    "price": price
                }
        
        for ticker in sellOrders.keys():
            self.__stockTracker.pop(ticker)
        
        return sellOrders

    def establishChoices(self) -> List:
        choices = []
        for ticker in self.__assets:
            asset: AssetDataPolars = self.__assets[ticker]
            aidx = self.__assetdateIdx[ticker]
            
            curDate = asset.adjClosePrice["Date"].item(aidx)
            curAdjPrice = asset.adjClosePrice["AdjClose"].item(aidx)
            curPrice = asset.shareprice["Close"].item(aidx)
            mMonthAdjPrice = asset.adjClosePrice["AdjClose"].item(aidx - 21)
            curVolume = asset.volume["Volume"].item(aidx)
            maxVolume = asset.volume["Volume"].slice(aidx - 20,21).max()
            curEPSIdx = DPl(asset.financials_quarterly, dateCol="fiscalDateEnding").getNextLowerOrEqualIndex(curDate)
            surprice = asset.financials_quarterly["surprise"].item(curEPSIdx)
            #lastEPS = asset.financials_annually["reportedEPS"].item(curEPSIdx-1)
            if surprice is not None and surprice <0.1:
                continue
            if (curAdjPrice < mMonthAdjPrice * 0.95 and
                curVolume == maxVolume):
                choices.append(ticker)
            
        return choices
    
    def buyOrders(self) -> Dict:
        buyOrders = {}

        # Select buy choices
        buyChoices: List = self.establishChoices()
        if buyChoices == []:
            return buyOrders
        
        buyChoices = shuffle(buyChoices)
        buyChoices = buyChoices[:self.__num_choices - len(self.__portfolio.positions.keys())]

        # Divide cash equally among selected stocks
        cashPerStock = self.__portfolio.cash / len(buyChoices)

        for topTicker in buyChoices:
            asset: AssetDataPolars = self.__assets[topTicker]
            price = asset.adjClosePrice["AdjClose"].item(self.__assetdateIdx[topTicker])
            if price < 0.00001:
                print("Adjusted Close Price is Negative. Skipping.")
                continue
            quantity = np.floor((cashPerStock) / (price + ActionCost().buy(price)))
            if abs(quantity) > 0.01:
                buyOrders[topTicker] = {
                    "quantity": quantity,
                    "price": price
                }
        
        for ticker in buyOrders.keys():
            self.__stockTracker[ticker] = {
                "boughtPrice": buyOrders[ticker]["price"],
                "boughtDateIdx": self.__assetdateIdx[ticker]
            }

        return buyOrders

    def apply(self,
              assets: Dict[str, AssetDataPolars], 
              currentDate: pd.Timestamp, 
              assetdateIdx: Dict[str, int] = {}):
        self.__assetdateIdx = assetdateIdx
        
        sellOrders = self.sellOrders()
        self.sell(sellOrders, self.__portfolio, currentDate)

        if self.__portfolio.positions.keys().__len__ == self.__num_choices:
            return
        
        if not self.__portfolio.valueOverTime == [] \
            and self.__portfolio.cash < self.__cashthreshhold * self.__portfolio.valueOverTime[-1][1] \
            and not sellOrders:
            return  # Do not buy if positions are not empty and no assets were sold.

        buyOrders = self.buyOrders()
        self.buy(buyOrders, self.__portfolio, currentDate)