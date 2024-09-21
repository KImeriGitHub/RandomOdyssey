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

    def apply(self, assets: Dict[str, AssetData], portfolio: Portfolio, current_time: pd.Timestamp):
        # Calculate the start date based on num_months
        start_date = current_time - pd.DateOffset(months=self.num_months)

        # List to store analysis results
        analysis_results = []

        for ticker, asset in assets.items():
            # Get price data for the specified period
            timeintervalIdx = (asset.shareprice.index >= start_date-pd.Timedelta(hours=18)) & \
                              (asset.shareprice.index < current_time+pd.Timedelta(hours=18))
            price_data: pd.DataFrame = asset.shareprice[timeintervalIdx]
            if price_data.empty:
                continue

            # Prepare data for linear regression
            price_data = price_data.resample('B').mean().dropna()  # Resample to business days
            X = np.arange(len(price_data)).reshape(-1, 1)
            y = price_data['Close'].values

            # Perform linear regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            slope = model.coef_[0]
            residuals = y - y_pred
            variance = np.var(residuals)

            # Store results
            analysis_results.append({
                'Ticker': ticker,
                'Slope': slope,
                'Variance': variance
            })

        if not analysis_results:
            print("No assets available for analysis.")
            return

        # Convert results to DataFrame
        results_df = pd.DataFrame(analysis_results)

        # Rank by highest slope and lowest variance
        results_df['Score'] = results_df['Slope'] / results_df['Variance']
        results_df.sort_values(by='Score', ascending=False, inplace=True)

        # Select top choices
        top_choices = results_df.head(self.num_choices)

        # Divide cash equally among selected stocks
        cash_per_stock = portfolio.cash / len(top_choices)

        for _, row in top_choices.iterrows():
            ticker = row['Ticker']
            asset = assets[ticker]
            timeintervalIdx = (asset.shareprice.index >= current_time-pd.Timedelta(hours=18)) & \
                              (asset.shareprice.index < current_time+pd.Timedelta(hours=18))
            price_data = asset.shareprice[timeintervalIdx]
            if not price_data.empty:
                price: float = float(price_data['Close'].iloc[0])
                quantity = np.floor((cash_per_stock - ActionCost.buy(cash_per_stock)) / price)
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
            timeintervalIdx = (asset.shareprice.index >= current_time-pd.Timedelta(hours=18)) & \
                              (asset.shareprice.index < current_time+pd.Timedelta(hours=18))
            price_data = asset.shareprice[timeintervalIdx]
            if not price_data.empty:
                price: float = float(price_data['Close'].iloc[0])
                sellOrders[boughtTicker] = [portfolio.positions[boughtTicker], price]

        for ticker in sellOrders.keys():
            portfolio.sell(ticker, sellOrders[ticker][0], sellOrders[ticker][1])
            print(f"Sold {sellOrders[ticker][0]} shares of {ticker} at {sellOrders[ticker][1]}.")