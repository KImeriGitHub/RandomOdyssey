# src/simulation/LinearAscendStrategy.py

from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class StratLinearAscend(IStrategy):
    def __init__(self, num_months: int = 2, num_choices: int = 1):
        self.num_months = num_months
        self.num_choices = num_choices

    def apply(self, assets: Dict[str, AssetData], portfolio: Portfolio, current_time: pd.Timestamp, assetdateIdx: Dict[str, int] = None):
        # Calculate the start date based on num_months
        start_date = current_time - pd.DateOffset(months=self.num_months)

        # List to store analysis results
        analysis_results = []

        for ticker, asset in assets.items():
            # Get price data for the specified period
            timeintervalIdx = (asset.shareprice.index >= start_date-pd.Timedelta(hours=12)) & \
                              (asset.shareprice.index < current_time+pd.Timedelta(hours=12))
            price_data: pd.DataFrame = asset.shareprice[timeintervalIdx]
            if price_data.empty:
                continue

            # Prepare data for linear regression
            price_data = price_data.resample('B').mean().dropna()  # Resample to business days
            x = np.arange(len(price_data))

            # Dependent variable: 'Close' prices
            y = price_data['Close'].values

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
            residuals = np.maximum(y-y_pred,0)
            variance = np.var(residuals)

            # Store results
            analysis_results.append({
                'Ticker': ticker,
                'Slope': m,
                'Variance': variance
            })

        if not analysis_results:
            print("No assets available for analysis.")
            return

        # Convert results to DataFrame
        results_df = pd.DataFrame(analysis_results)

        # Rank by highest slope and lowest variance
        results_df['Rankslope'] = results_df['Slope'].rank(ascending=False)
        results_df['Rankvar'] = results_df['Variance'].rank(ascending=True)
        results_df['Score'] = results_df['Rankslope'] + results_df['Rankvar']
        results_df.sort_values(by='Score', ascending=True, inplace=True)

        # Select top choices
        top_choices = results_df.head(self.num_choices)

        # Divide cash equally among selected stocks
        cash_per_stock = portfolio.cash / len(top_choices)

        for _, row in top_choices.iterrows():
            if 'Value' in portfolio.history.columns \
                and not portfolio.history['Value'].empty \
                and portfolio.cash/portfolio.history['Value'].iloc[-1] < 0.1:
                continue

            ticker = row['Ticker']
            asset = assets[ticker]
            timeintervalIdx = (asset.shareprice.index >= current_time-pd.Timedelta(hours=12)) & \
                              (asset.shareprice.index < current_time+pd.Timedelta(hours=12))
            price_data = asset.shareprice[timeintervalIdx]
            if not price_data.empty:
                price: float = float(price_data['Close'].iloc[-1])
                if price < 0.00001:
                    continue
                quantity = np.floor((cash_per_stock - ActionCost().buy(cash_per_stock)) / price)
                if abs(quantity)>0.0001:
                    portfolio.buy(ticker, quantity, price)
                    print(f"Bought {quantity} shares of {ticker} at {price}")
            else:
                print(f"No price data for {ticker} on {current_time}")

        sellOrders = {}
        for boughtTicker in portfolio.positions.keys():
            if top_choices["Ticker"].isin([boughtTicker]).any():
                continue

            asset = assets[boughtTicker]
            timeintervalIdx = (asset.shareprice.index >= current_time-pd.Timedelta(hours=12)) & \
                              (asset.shareprice.index < current_time+pd.Timedelta(hours=12))
            price_data = asset.shareprice[timeintervalIdx]
            if not price_data.empty:
                price: float = float(price_data['Close'].iloc[-1])
                if price < 0.00001:
                    continue
                sellOrders[boughtTicker] = [portfolio.positions[boughtTicker], price]

        for ticker in sellOrders.keys():
            portfolio.sell(ticker, sellOrders[ticker][0], sellOrders[ticker][1])
            print(f"Sold {sellOrders[ticker][0]} shares of {ticker} at {sellOrders[ticker][1]}.")