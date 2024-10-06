from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DFTO
from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

class StratLinearAscend(IStrategy):
    __stoplossRatio = 0.95

    def __init__(self,
                 num_months: int = 2, 
                 num_choices: int = 1,
                 ):
        self.num_months: int = num_months
        self.num_choices: int = num_choices

        self.__assets: Dict[str, AssetData] = {}
        self.__portfolio: Portfolio = None

        self.__stoplossLimit: Dict[str, float] = {}
        self.__blacklist: Dict[str, pd.Timestamp] = {}

        self.__currentDate: pd.Timestamp = pd.Timestamp(None)
        self.__assetdateIdx: Dict[str, int] = {}

    def sellOrders(self) -> Dict:
        sellOrders = {}
        for boughtTicker in self.__portfolio.positions.keys():
            asset: AssetData = self.__assets[boughtTicker]
            price_data = asset.shareprice.iloc[self.__assetdateIdx[boughtTicker]]

            if not price_data.empty:
                price: float = float(price_data['Close'])
                if price < 0.00001:
                    print("Weird price.")
                    continue

                if price <= self.__stoplossLimit[boughtTicker]:
                    sellOrders[boughtTicker] = {
                        "quantity": self.__portfolio.positions[boughtTicker],
                        "price": price
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

    def preAnalyze(self) -> pd.DataFrame:
        modAssetList: List[str] = []
        priceData: float = 0.0
        for ticker in self.__assets.keys():
            asset: AssetData = self.__assets[ticker]
            priceData = asset.shareprice.iloc[self.__assetdateIdx[ticker]]['Close']
            if priceData > 10.0:
                modAssetList.append(ticker)

        analysis_results = []
        startDate = self.__currentDate - pd.DateOffset(months=self.num_months)
        for ticker in modAssetList:
            asset: AssetData = self.__assets[ticker]
            priceData: pd.DataFrame = DFTO(asset.shareprice).inbetween(startDate, self.__currentDate, pd.Timedelta(hours=18))
            if priceData.empty:
                continue

            # Prepare data for linear regression
            priceData = priceData.resample('B').mean().dropna()  # Resample to business days

            # Store results
            analysis_results.append(self.curveAnalysis(priceData['Close'].values, ticker))

        if not analysis_results:
            print("No assets available for analysis.")
            return
        
        return pd.DataFrame(analysis_results)

    def rankAssets(self) -> pd.DataFrame:
        results_df: pd.DataFrame = self.preAnalyze()

        # Calculate the 75% quantile of the 'Slope' column
        quant = results_df['Slope'].quantile(0.90)
        # Initialize 'Rankslope' by assigning rank 1 to values above the  quantile
        results_df['Rankslope'] = np.where(results_df['Slope'] > quant, 1, np.nan)
        # Create a mask for values at or below the quantile
        mask = results_df['Slope'] <= quant
        # Rank the remaining values in descending order, starting from rank 2
        results_df.loc[mask, 'Rankslope'] = (
            results_df.loc[mask, 'Slope'].rank(ascending=False, method='dense') + 1
        )
        results_df['Rankvar'] = results_df['Variance'].rank(ascending=True)
        results_df['Score'] = results_df['Rankslope'] + results_df['Rankvar']
        results_df.sort_values(by='Score', ascending=True, inplace=True)

        # Select top choices
        return results_df.head(self.num_choices)
    
    def buyOrders(self) -> Dict:
        buyOrders = {}

        # Select top choices
        top_choices: pd.DataFrame = self.rankAssets()

        # Divide cash equally among selected stocks
        cashPerStock = self.__portfolio.cash / len(top_choices)

        for _, row in top_choices.iterrows():
            ticker: str = row['Ticker']
            asset: AssetData = self.__assets[ticker]
            price_data: pd.DataFrame = asset.shareprice.iloc[self.__assetdateIdx[ticker]]
            if not price_data.empty:
                price: float = float(price_data['Close'])
                if price < 0.00001:
                    continue
                quantity = np.floor((cashPerStock) / (price + ActionCost().buy(price)))
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

        if len(self.__portfolio.positions.keys()) > 0 and not sellOrders:
            return  # Do not buy if positions are not empty and no assets were sold.

        buyOrders = self.buyOrders()
        self.buy(buyOrders)

        self.updateStoplossLimit()

    def curveAnalysis(self, priceArray, ticker: str) -> Dict:
        x = np.arange(len(priceArray))

        # Dependent variable: 'Close' prices
        y = priceArray

        # Fit line through the first data point and minimize residuals
        x0 = x[0]
        y0 = y[0]
        dx = x - x0
        dy = y - y0

        denominator = np.sum(dx * dx)
        if denominator == 0:
            raise ValueError("Denominator is zero; all x values are the same.")
        
        # Calculate the slope (m) and intercept (c)
        m = np.sum(dx * dy) / denominator
        c = y0 - m * x0

        # Predict y-values using the fitted line
        y_pred = m * x + c

        # Calculate residuals and variance
        residuals = y-y_pred
        
        mean_y_pred = np.mean(y_pred)
        if mean_y_pred == 0:
            mean_y_pred = np.finfo(float).eps  # Smallest representable float
        variance = np.var(residuals / mean_y_pred)

        return {
            'Ticker': ticker,
            'Slope': m,
            'Variance': variance
        }