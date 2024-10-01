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
    def __init__(self, num_months: int = 2, num_choices: int = 1):
        self.num_months = num_months
        self.num_choices = num_choices

        self.stoplossLimit: Dict[str, float] = {}  # Ticker symbol to limit
        self.blacklist: Dict[str, pd.Timestamp]

    def apply(self, assets: Dict[str, AssetData], portfolio: Portfolio, currentDate: pd.Timestamp, assetdateIdx: Dict[str, int] = None):
        # CHECK SELL
        sellOrders = {}
        for boughtTicker in portfolio.positions.keys():
            asset = assets[boughtTicker]
            timeOps = DFTO(asset.shareprice)
            price_data = asset.shareprice.iloc[assetdateIdx[boughtTicker]]

            if not price_data.empty:
                price: float = float(price_data['Close'])

                if price < 0.00001:
                    continue

                if price <= self.stoplossLimit[boughtTicker]:
                    sellOrders[boughtTicker] = [portfolio.positions[boughtTicker], price]

        # Sell
        for ticker in sellOrders.keys():
            portfolio.sell(ticker, sellOrders[ticker][0], sellOrders[ticker][1], currentDate)
            self.stoplossLimit.pop(ticker)
            print(f"Sold {sellOrders[ticker][0]} shares of {ticker} at {sellOrders[ticker][1]}.")

        # UPDATE STOPLOSS LIMIT
        # Calculate the start date based on num_months
        start_date = currentDate - pd.DateOffset(months=self.num_months)
        for portticker in portfolio.positions.keys():
            asset = assets[portticker]
            timeOps = DFTO(asset.shareprice)
            # timeintervalIdx = (asset.shareprice.index >= start_date - pd.Timedelta(hours=12)) & \
            #                   (asset.shareprice.index < current_time + pd.Timedelta(hours=12))
            # price_data: pd.DataFrame = asset.shareprice[timeintervalIdx]
            price_data = timeOps.inbetween(start_date,currentDate,pd.Timedelta(hours=12))
            price_data = price_data.resample('B').mean().dropna() # Resample to business days
            price_data = price_data['Close'].values
            self.stoplossLimit[portticker] =  price_data[-1]*0.95 \
                    if price_data[-1]*0.95 > self.stoplossLimit[portticker] \
                    else self.stoplossLimit[portticker]

        if len(portfolio.positions.keys())>0 and not sellOrders:
            return # Do not buy if empty

        # BUYING
        # List to store analysis results
        analysis_results = []

        for ticker, asset in assets.items():
            # Get price data for the specified period
            timeOps = DFTO(asset.shareprice)
            #timeintervalIdx = (asset.shareprice.index >= start_date-pd.Timedelta(hours=12)) & \
            #                  (asset.shareprice.index < current_time+pd.Timedelta(hours=12))
            #price_data: pd.DataFrame = asset.shareprice[timeintervalIdx]
            price_data = timeOps.inbetween(start_date, currentDate,pd.Timedelta(hours=12))
            if price_data.empty:
                continue

            # Prepare data for linear regression
            price_data = price_data.resample('B').mean().dropna()  # Resample to business days

            # Store results
            analysis_results.append(self.curveAnalysis(price_data['Close'].values, ticker))

        if not analysis_results:
            print("No assets available for analysis.")
            return

        # Convert results to DataFrame
        results_df = pd.DataFrame(analysis_results)

        # Calculate the 75% quantile of the 'Slope' column
        quant = results_df['Slope'].quantile(0.60)

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
        top_choices = results_df.head(self.num_choices)

        # Divide cash equally among selected stocks
        cash_per_stock = portfolio.cash / len(top_choices)

        for _, row in top_choices.iterrows():
            ticker = row['Ticker']
            asset = assets[ticker]
            timeOps = DFTO(asset.shareprice)
            #timeintervalIdx = (asset.shareprice.index >= current_time-pd.Timedelta(hours=12)) & \
            #                  (asset.shareprice.index < current_time+pd.Timedelta(hours=12))
            #price_data = asset.shareprice[timeintervalIdx]
            price_data = asset.shareprice.iloc[assetdateIdx[ticker]]
            if not price_data.empty:
                price: float = float(price_data['Close'])
                if price < 0.00001:
                    continue
                quantity = np.floor((cash_per_stock - ActionCost().buy(cash_per_stock)) / price)
                if abs(quantity) > 0.0001:
                    portfolio.buy(ticker, quantity, price, currentDate)
                    self.stoplossLimit[ticker] = price*0.95
                    print(f"Bought {quantity} shares of {ticker} at {price}")
            else:
                print(f"No price data for {ticker} on {currentDate}")

    def curveAnalysis(self, priceArray, ticker: str):
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
        #variance = np.var(np.maximum(residuals,0))
        variance = np.var(residuals/np.mean(y_pred))

        return {
            'Ticker': ticker,
            'Slope': m,
            'Variance': variance
        }