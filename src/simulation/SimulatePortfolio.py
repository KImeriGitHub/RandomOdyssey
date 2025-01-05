from typing import List, Dict
from src.common.Portfolio import Portfolio
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars
from src.simulation.ISimulation import ISimulation
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
import pandas as pd
import polars as pl
import numpy as np
import datetime
import warnings
import line_profiler

class SimulatePortfolio(ISimulation):
    def __init__(self, 
                strategy: IStrategy,
                assets: Dict[str, AssetDataPolars], 
                portfolio: Portfolio, 
                startDate: pd.Timestamp, 
                endDate: pd.Timestamp):
    
        self.portfolio = portfolio
        self.strategy = strategy
        self.assets = assets
        
        self.startDate = startDate
        self.endDate = endDate

    def __checkAssetSetupIdx(self) -> Dict:
        # FOR FASTER RUN: Establish index in dataframe to start date
        assetdateIdx = {}
        discardedAsset = []
        for ticker, asset in self.assets.items():
            if asset.adjClosePrice.select(pl.col("Date").last()).item() < self.startDate:
                warnings.warn(f"Asset {ticker} history not old enough or startDate ({self.startDate}) too far back. It is discarded.")
                discardedAsset.append(ticker)
                continue

            maxDays=5
            if self.startDate + pd.Timedelta(days=maxDays) > self.endDate:
                raise ValueError(f"End Date ({self.endDate}) must be {maxDays} after Start Date {self.startDate}.")
            for i in range(maxDays+1):
                dateToCheck = self.startDate + pd.Timedelta(days=i)
                idx = DPl(asset.shareprice).getIndex(dateToCheck, pd.Timedelta(days=0.7))
                if idx != -1:
                    assetdateIdx[ticker] = idx
                    break
                else:
                    message = f"Asset {ticker} start index could not be established on date ({dateToCheck})."
                    if i < maxDays:
                        message += " Checking next day."
                    else:
                        message += " It is discarded."
                        assetdateIdx.pop(ticker)
                        discardedAsset.append(ticker)
                    warnings.warn(message)

        for ticker in discardedAsset:
            self.assets.pop(ticker)

        return assetdateIdx

    def run(self):
        assetdateIdx = self.__checkAssetSetupIdx()

        # Init Calculation of Asset Prices
        asset_prices = {}
        price_data = pl.DataFrame(None)
        for ticker, asset in self.assets.items():
            price_data = asset.adjClosePrice['AdjClose']
            asset_prices[ticker] = price_data.item(assetdateIdx[ticker])

        # Main Loop
        dates = pd.date_range(self.startDate, self.endDate, freq='B') # 'B' for business days
        for date in dates:
            # Apply the strategy
            self.strategy.apply(
                assets = self.assets,
                currentDate = date, 
                assetdateIdx = assetdateIdx)

            # Advance assetdateIdx
            for ticker, asset in self.assets.items():
                if assetdateIdx[ticker] > asset.adjClosePrice['Date'].len():
                    warnings.warn(f"Out of Bound access for {ticker}. Skipped.")
                    continue
                assetDate: datetime.datetime = asset.adjClosePrice['Date'].item(assetdateIdx[ticker])
                if (assetDate >= date - pd.Timedelta(hours=18)) & (assetDate < date + pd.Timedelta(hours=18)):
                    asset_prices[ticker] = asset.adjClosePrice['AdjClose'].item(assetdateIdx[ticker])
                    assetdateIdx[ticker] += 1

            # Update portfolio value
            self.portfolio.updateValue(date, asset_prices)

        del assetdateIdx