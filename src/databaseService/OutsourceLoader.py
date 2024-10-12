import yfinance as yf
import pandas as pd
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
            self._load_from_yfinance(assetData, ticker, raise_errors=True)
        
        return assetData

    def _load_from_yfinance(self, assetData: AssetData, tickerHandle: str, raise_errors=False):
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

        fullSharePrice = stock.history(period="max", raise_errors = raise_errors)
        try:
            assetData.shareprice = fullSharePrice[["Open", "High", "Low", "Close", "Adj Close"]]

            assetData.volume = fullSharePrice["Volume"]  #think on how to deal with nan results

            assetData.dividends = fullSharePrice["Dividends"].dropna()
            assetData.dividends = assetData.dividends[(assetData.dividends > 0.000001) | (assetData.dividends < -0.000001)]

            assetData.splits = fullSharePrice["Stock Splits"].dropna()
            assetData.splits = assetData.splits[(assetData.splits > 0.000001) | (assetData.splits < -0.000001)]
        except:
            warnings.warn("Failed to retrieve Price History for ticker: " + tickerHandle)

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
