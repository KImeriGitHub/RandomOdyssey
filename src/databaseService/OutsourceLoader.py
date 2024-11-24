import yfinance as yf
import pandas as pd
import numpy as np
import warnings

from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService


class OutsourceLoader:
    outsourceOperator: str  # only yfinance supported

    def __init__(self, outsourceOperator: str):
        self.outsourceOperator = outsourceOperator
        
        if self.outsourceOperator != "yfinance":
            raise NotImplementedError("Only yfinance supported as of now.")
    
    def load(self, ticker: str) -> AssetData:
        assetData: AssetData = AssetDataService.defaultInstance()
        if self.outsourceOperator == "yfinance":
            self._load_from_yfinance(assetData, ticker)
        
        return assetData

    def _load_from_yfinance(self, assetData: AssetData, tickerHandle: str):
        stock = yf.Ticker(tickerHandle)

        # The saved ticker symbol is not the tickerHandle.
        assetData.ticker = stock.ticker

        try:
            assetData.isin = stock.isin
        except:
            warnings.warn("Failed to retrieve ISIN for ticker: " + tickerHandle)

        try:
            assetData.about = stock.info
        except:
            warnings.warn("Failed to retrieve INFO for ticker: " + tickerHandle)

        fullSharePrice: pd.DataFrame = yf.download(tickerHandle, period="max")
        if not fullSharePrice.empty:
            assetData.shareprice = fullSharePrice[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            assetData.shareprice.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            assetData.volume = assetData.shareprice["Volume"]
            assetData.dividends = stock.dividends
            assetData.splits = stock.splits

            assetData.adjClosePrice = assetData.shareprice["Adj Close"]
            self._fill_NAN_to_BDays(assetData.adjClosePrice)
        else:
            raise ValueError("Failed to retrieve Price History for ticker: " + tickerHandle)

        try:
            fullFinancials = stock.quarterly_financials
            fullFinancials = fullFinancials.T
            fullFinancials.index.name = 'Date'
            fullFinancials.index = pd.to_datetime(fullFinancials.index)

            assetData.revenue = fullFinancials["Total Revenue"]
            assetData.EBITDA = fullFinancials["EBITDA"]
            assetData.basicEPS = fullFinancials["Basic EPS"]
        except:
            warnings.warn("Failed to retrieve Financial Data for ticker: " + tickerHandle)

    def _fill_NAN_to_BDays(self, s: pd.Series):
        # PRE: s has index as dates. They are sorted.
        # POST: Completes the dates to business days
        #   Adds to nan values a normal random number with mean of the neighbouring prices and sigma half their difference.

        # Ensure the index is a DateTimeIndex
        s.index = pd.to_datetime(s.index)

        # Generate a date range covering all business days between the earliest and latest dates
        date_range = pd.bdate_range(start=s.index.min(), end=s.index.max())

        # Reindex the series to include all business days
        s = s.reindex(date_range)

        # Initialize a list to store indices of missing values
        missing_indices = s[s.isna()].index

        # Iterate over the missing dates to fill them
        for date in missing_indices:
            # Get the position of the current missing date
            idx = s.index.get_loc(date)

            # Find previous valid price
            prev_idx = idx - 1
            while prev_idx >= 0 and pd.isna(s.iloc[prev_idx]):
                prev_idx -= 1

            # Find next valid price
            next_idx = idx + 1
            while next_idx < len(s) and pd.isna(s.iloc[next_idx]):
                next_idx += 1

            # If both previous and next prices are found
            if prev_idx >= 0 and next_idx < len(s):
                prev_price = s.iloc[prev_idx]
                next_price = s.iloc[next_idx]
                mean = (prev_price + next_price) / 2
                sigma = abs(next_price - prev_price) / 2

                # Generate a random value from the normal distribution
                random_value = np.random.normal(mean, sigma)

                # Assign the random value to the missing date
                s.iloc[idx] = random_value
            else:
                # If only one neighbor is available, fill with that price
                if prev_idx >= 0:
                    s.iloc[idx] = s.iloc[prev_idx]
                elif next_idx < len(s):
                    s.iloc[idx] = s.iloc[next_idx]
                else:
                    # If neither neighbor is available, leave as 0
                    s.iloc[idx] = 0
