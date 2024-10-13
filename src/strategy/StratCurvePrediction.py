from src.mathTools.CurveAnalysis import CurveAnalysis
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DFTO
from src.predictionModule.CurveML import CurveML
from typing import Dict, List
import pandas as pd
import numpy as np

class StratCurvePrediction(IStrategy):
    __stoplossRatio = 0.95
    __cashthreshhold = 0.2

    def __init__(self,
                 num_months: int = 2,
                 modelPath: str = "",
                 modelName: str = ""):
        self.num_months: int = num_months

        self.__assets: Dict[str, AssetData] = {}
        self.__portfolio: Portfolio = None

        self.__stoplossLimit: Dict[str, float] = {}
        self.__blacklist: Dict[str, pd.Timestamp] = {}

        self.__currentDate: pd.Timestamp = pd.Timestamp(None)
        self.__assetdateIdx: Dict[str, int] = {}

        self.__curveML = CurveML(self.__assets, pd.Timestamp(None), pd.Timestamp(None))
        self.__curveML.loadModel(modelPath, modelName)

    def sellOrders(self) -> Dict:
        sellOrders = {}
        for boughtTicker in self.__portfolio.positions.keys():
            asset: AssetData = self.__assets[boughtTicker]
            priceData = asset.shareprice.iloc[(self.__assetdateIdx[boughtTicker]-21):(self.__assetdateIdx[boughtTicker]+1)]["Close"]
            priceData = priceData.resample('B').mean().dropna()
            predictedPrice = self.__curveML.predictNextPrices(priceData, boughtTicker, 1)

            currentPrice=priceData.values[-1]
            if currentPrice <= self.__stoplossLimit[boughtTicker] \
                or predictedPrice[-1] <= self.__stoplossLimit[boughtTicker]:
                sellOrders[boughtTicker] = {
                    "quantity": self.__portfolio.positions[boughtTicker],
                    "price": currentPrice
                }

        return sellOrders

    def sell(self, sellOrders: Dict):
        if not sellOrders:
            return
        # Sell
        for ticker in sellOrders.keys():
            quantity = sellOrders[ticker]['quantity']
            price = sellOrders[ticker]['price']
            self.__portfolio.sell(ticker, quantity, price, self.__currentDate)
            self.__stoplossLimit.pop(ticker)
            print(f"Sold {quantity} shares of {ticker} at {price} on date: {self.__currentDate}.")

    def updateStoplossLimit(self):
        for portticker in self.__portfolio.positions.keys():
            asset: AssetData = self.__assets[portticker]
            price_data = asset.shareprice.iloc[self.__assetdateIdx[portticker]]
            price_data = price_data['Close']
            if price_data * self.__stoplossRatio > self.__stoplossLimit[portticker]:
                self.__stoplossLimit[portticker] =  price_data * self.__stoplossRatio
            else:
                self.__stoplossLimit[portticker] = self.__stoplossLimit[portticker]

    def preAnalyze(self) -> List:
        analysis_results = []
        startDate = self.__currentDate - pd.DateOffset(months=self.num_months)
        for ticker, asset in self.__assets.items():
            priceData: pd.DataFrame = DFTO(asset.shareprice).inbetween(startDate, self.__currentDate, pd.Timedelta(hours=18))
            if priceData.empty:
                continue

            # Prepare data for linear regression
            priceData = priceData["Close"]
            priceData = priceData.resample('B').mean().dropna()  # Resample to business days

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
        analysis_results = self.preAnalyze()
        maxprod = 1
        topChoice = ""
        for tickerResult in analysis_results:
            prod = np.prod(tickerResult["predicted ratio"])
            if prod > maxprod:
                topChoice = tickerResult["ticker"]
                maxprod=prod

        if topChoice == "":
            return buyOrders

        ticker: str = topChoice
        asset: AssetData = self.__assets[ticker]
        priceData: pd.DataFrame = asset.shareprice.iloc[self.__assetdateIdx[ticker]]
        if not priceData.empty:
            price: float = float(priceData['Close'])
            quantity = np.floor((self.__portfolio.cash) / (price + ActionCost().buy(price)))
            if abs(quantity) > 0.00001:
                buyOrders[ticker] = {
                    "quantity": quantity,
                    "price": price
                }
        else:
            print(f"No price data for {ticker} on {self.__currentDate}")

        return buyOrders

    def buy(self, buyOrders: Dict):
        if not buyOrders:
            return
        for ticker in buyOrders.keys():
            quantity = buyOrders[ticker]['quantity']
            price = buyOrders[ticker]['price']
            self.__portfolio.buy(ticker, quantity, price, self.__currentDate)
            self.__stoplossLimit[ticker] = price * self.__stoplossRatio
            print(f"Bought {quantity} shares of {ticker} at {price} on Date: {self.__currentDate}.")

    def apply(self,
              assets: Dict[str, AssetData], 
              portfolio: Portfolio, 
              currentDate: pd.Timestamp, 
              assetdateIdx: Dict[str, int] = {}):
        
        self.__assets = assets
        self.__portfolio = portfolio
        self.__currentDate = currentDate
        self.__assetdateIdx = assetdateIdx

        sellOrders = self.sellOrders()
        self.sell(sellOrders)

        self.updateStoplossLimit()

        if not len(self.__portfolio.valueOverTime) == 0 \
            and portfolio.cash / portfolio.valueOverTime[-1][1] < self.__cashthreshhold:
            return  # stop if cash is smaller than threshold

        buyOrders = self.buyOrders()
        self.buy(buyOrders)

        self.updateStoplossLimit()