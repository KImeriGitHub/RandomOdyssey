import yfinance as yf
import pandas as pd

from src.common.AssetData import AssetData

class OutsourceLoader:
    outsourceOperator: str  # only yfinance supported

    def __init__(self, outsourceOperator: str):
        self.outsourceOperator = outsourceOperator
        
        if self.outsourceOperator != "yfinance":
            raise NotImplementedError("Only yfinance supported as of now.")
    
    def load(self, ticker: str) -> AssetData:
        if self.outsourceOperator == "yfinance":
            assetData: AssetData = AssetData(ticker = ticker, 
                                isin = "", 
                                shareprice = None,
                                volume = None,
                                dividends = None,
                                splits = None,
                                about = None,
                                revenue = None,
                                EBITDA = None,
                                basicEPS = None)
            self._load_from_yfinance(assetData, ticker)
            return assetData

    def _load_from_yfinance(self, assetData: AssetData, ticker: str):
        stock = yf.Ticker(ticker)

        try:
            assetData.isin = stock.isin
        except:
            assetData.isin = None
            return

        assetData.about = stock.info
        fullSharePrice = stock.history(period="max")
        assetData.shareprice = fullSharePrice[["Open", "High", "Low", "Close"]]
        assetData.volume = fullSharePrice[["Volume"]]
        assetData.dividends = fullSharePrice["Dividends"]
        assetData.splits = fullSharePrice["Stock Splits"]

        fullFinancials = stock.quarterly_financials
        fullFinancials = fullFinancials.T
        fullFinancials.index.name = 'Date'
        fullFinancials.index = pd.to_datetime(fullFinancials.index)

        assetData.revenue = fullFinancials["Total Revenue"]
        assetData.EBITDA = fullFinancials["EBITDA"]
        assetData.basicEPS = fullFinancials["Basic EPS"]
