# src/simulation/LinearAscendStrategy.py

from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
from typing import Dict, List
import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class StratLinearAscend(IStrategy):
    def __init__(self, num_months: int = 2, num_choices: int = 1):
        self.num_months = num_months
        self.num_choices = num_choices
        self.has_bought = False

    def apply(self, assets: Dict[str, AssetData], portfolio: Portfolio, current_time: datetime.datetime):
        if self.has_bought:
            return

        # Calculate the start date based on num_months
        start_date = current_time - pd.DateOffset(months=self.num_months)
        
        # List to store analysis results
        analysis_results = []

        for ticker, asset in assets.items():
            # Get price data for the specified period
            price_data = asset.shareprice[(asset.shareprice.index >= start_date) & (asset.shareprice.index <= current_time)]
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
            price_data = asset.shareprice.loc[asset.shareprice.index == current_time]
            if not price_data.empty:
                price = price_data['Close'].values[0]
                quantity = cash_per_stock / price
                portfolio.buy(ticker, quantity, price)
                print(f"Bought {quantity} shares of {ticker} at {price}")
            else:
                print(f"No price data for {ticker} on {current_time}")

        self.has_bought = True
