from src.mathTools.CurveAnalysis import CurveAnalysis
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPandas as DFTO
from src.predictionModule.CurveML import CurveML
from typing import Dict, List
import pandas as pd
import polars as pl
import numpy as np

class StratCurvePrediction(IStrategy):
    __stoplossRatio = 0.92
    __cashthreshhold = 0.2

    def __init__(self,
                 num_months: int = 2,
                 modelPath: str = "",
                 modelName: str = ""):
        self.num_months: int = num_months

        self.__assets: Dict[str, AssetDataPolars] = {}
        self.__portfolio: Portfolio = None

        self.__stoplossLimit: Dict[str, float] = {}
        self.__blacklist: Dict[str, pd.Timestamp] = {}

        self.__assetdateIdx: Dict[str, int] = {}

        self.__curveML = CurveML(self.__assets, pd.Timestamp(None), pd.Timestamp(None))
        self.__curveML.loadModel(modelPath, modelName)

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
        startDateIdxDiff = self.num_months*21
        for ticker in self.__assets:
            asset: AssetDataPolars = self.__assets[ticker]
            priceData: pl.DataFrame = asset.adjClosePrice["AdjClose"].slice(self.__assetdateIdx[ticker]-startDateIdxDiff, startDateIdxDiff+1)

            # Store results
            predictedPrice = self.__curveML.predictNextPrices(priceData, ticker, 1)
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
            price_data: pl.DataFrame = asset.shareprice["Close"].item(self.__assetdateIdx[topTicker])
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
              portfolio: Portfolio, 
              currentDate: pd.Timestamp, 
              assetdateIdx: Dict[str, int] = {}):
        
        self.__assets = assets
        self.__portfolio = portfolio
        self.__assetdateIdx = assetdateIdx

        sellOrders = self.sellOrders()
        self.sell(sellOrders, portfolio, currentDate, self.__stoplossLimit)

        self.updateStoplossLimit()

        if not self.__portfolio.valueOverTime == [] \
            and self.__portfolio.cash/self.__portfolio.valueOverTime[-1][1] < self.__cashthreshhold \
            and not sellOrders:
            return  # Do not buy if positions are not empty and no assets were sold.

        buyOrders = self.buyOrders()
        self.buy(buyOrders, portfolio, currentDate, self.__stoplossLimit)

        self.updateStoplossLimit()