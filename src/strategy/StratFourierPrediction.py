from src.mathTools.CurveAnalysis import CurveAnalysis
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPandas as DFTO
from src.predictionModule.FourierML import FourierML
from typing import Dict, List
import pandas as pd
import polars as pl
import numpy as np

class StratFourierPrediction(IStrategy):
    __stoplossRatio = 0.92
    __cashthreshhold = 0.2

    def __init__(self,
                 num_choices: int = 1,
                 modelPath: str = "",
                 modelName: str = ""):

        self.__num_choices: int = num_choices

        self.__assets: Dict[str, AssetDataPolars] = {}
        self.__portfolio: Portfolio = None

        self.__stoplossLimit: Dict[str, float] = {}
        self.__blacklist: Dict[str, pd.Timestamp] = {}

        self.__assetdateIdx: Dict[str, int] = {}

        self.__fourierML = FourierML(self.__assets, pd.Timestamp(None), pd.Timestamp(None))
        self.__fourierML.loadCNNModel(modelPath, modelName)

        self.__nextDayPrediction: Dict[str, float] = {}

    def sellOrders(self) -> Dict:
        sellOrders = {}
        for boughtTicker in self.__portfolio.positions.keys():
            asset: AssetDataPolars = self.__assets[boughtTicker]
            price = asset.adjClosePrice['AdjClose'].item(self.__assetdateIdx[boughtTicker])

            if price <= self.__stoplossLimit[boughtTicker]:
                sellOrders[boughtTicker] = {
                    "quantity": self.__portfolio.positions[boughtTicker],
                    "price": price
                }
        return sellOrders

    def updateStoplossLimit(self):
        for portticker in self.__portfolio.positions.keys():
            asset: AssetDataPolars = self.__assets[portticker]
            price_data = asset.adjClosePrice["AdjClose"].item(self.__assetdateIdx[portticker])
            if price_data * self.__stoplossRatio > self.__stoplossLimit[portticker]:
                self.__stoplossLimit[portticker] =  price_data * self.__stoplossRatio
            else:
                self.__stoplossLimit[portticker] = self.__stoplossLimit[portticker]

    def preAnalyze(self) -> List:
        analysis_results = []
        for ticker in self.__assets:
            asset: AssetDataPolars = self.__assets[ticker]
            aidx = self.__assetdateIdx[ticker]
            priceData: pl.DataFrame = asset.adjClosePrice["AdjClose"].slice(aidx-24 * 21, 24 * 21 +1).to_numpy()

            # Store results
            predictedPrice = self.__fourierML.predictNextPrices(priceData, ticker, 1)
            ratios = predictedPrice[1:] / predictedPrice[:-1]

            analysis_results.append({
                "ticker": ticker,
                "predicted ratio": ratios
            })

        if not analysis_results:
            print("No assets available for analysis!")
            return pd.DataFrame(None)
        
        return analysis_results
    
    def buyOrders(self) -> Dict:
        buyOrders = {}

        # Select top choices
        top_choices: pl.DataFrame = self.rankAssets()
        if top_choices is pl.DataFrame(None):
            return buyOrders

        # Divide cash equally among selected stocks
        cashPerStock = self.__portfolio.cash / len(top_choices)

        for topTicker in top_choices["Ticker"].to_list():
            asset: AssetDataPolars = self.__assets[topTicker]
            price_data: pl.DataFrame = asset.adjClosePrice["AdjClose"].item(self.__assetdateIdx[topTicker])
            price: float = float(price_data)
            if price < 0.00001:
                continue
            quantity = np.floor((cashPerStock) / (price + ActionCost().buy(price)))
            if abs(quantity) > 0.00001:
                buyOrders[topTicker] = {
                    "quantity": quantity,
                    "price": price
                }

        return buyOrders

    def apply(self,
              assets: Dict[str, AssetDataPolars], 
              currentDate: pd.Timestamp, 
              assetdateIdx: Dict[str, int] = {}):

        sellOrders = self.sellOrders()
        self.sell(sellOrders, self.__portfolio, currentDate, self.__stoplossLimit)

        self.updateStoplossLimit()

        if not self.__portfolio.valueOverTime == [] \
            and self.__portfolio.cash/self.__portfolio.valueOverTime[-1][1] < self.__cashthreshhold \
            and not sellOrders \
            or self.__portfolio.positions.keys().__len__ == self.num_choices:
            return  # Do not buy if positions are not empty and no assets were sold.

        buyOrders = self.buyOrders()
        self.buy(buyOrders, self.__portfolio, currentDate, self.__stoplossLimit)

        self.updateStoplossLimit()

        self.__nextDayPrediction = {}