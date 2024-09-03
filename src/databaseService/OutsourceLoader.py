import yfinance as yf
import pandas as pd

from src.common import AssetData

class OutsourceLoader:
    outsourceOperator: str  # only yfinance supported

    def __init__(self, outsourceOperator: str):
        self.outsourceOperator = outsourceOperator
        
        if self.outsourceOperator != "yfinance":
            raise NotImplementedError("Only yfinance supported as of now.")
    
    def load(self, assetData:AssetData, ticker: str):
        if self.outsourceOperator == "yfinance":
            self._load_from_yfinance(self, assetData, ticker)

    def _load_from_yfinance(self, assetData:AssetData, ticker: str):
        stock = yf.Ticker(ticker)

        try:
            assetData.isin = stock.isin
        except:
            assetData.isin = None
            return

        assetData.about = stock.info
        


        stock = yf.Ticker(ticker)
        self.stocks[ticker] = {
            "info": stock.info,
            "history": stock.history(period="max"),
            "actions": stock.actions,
            "dividends": stock.dividends,
            "splits": stock.splits,
        }
    
    def get_stock_info(self, ticker):
        return self.stocks.get(ticker, {}).get("info", {})
    
    def get_stock_history(self, ticker):
        return self.stocks.get(ticker, {}).get("history", pd.DataFrame())
    
    def get_stock_actions(self, ticker):
        return self.stocks.get(ticker, {}).get("actions", pd.DataFrame())
    
    def get_stock_dividends(self, ticker):
        return self.stocks.get(ticker, {}).get("dividends", pd.Series())
    
    def get_stock_splits(self, ticker):
        return self.stocks.get(ticker, {}).get("splits", pd.Series())
    
    def save_stock_data(self, ticker, filename):
        if ticker not in self.stocks:
            raise ValueError(f"Stock '{ticker}' not loaded.")
        
        stock_data = self.stocks[ticker]
        
        with pd.ExcelWriter(filename) as writer:
            stock_data['history'].to_excel(writer, sheet_name='History')
            stock_data['actions'].to_excel(writer, sheet_name='Actions')
            stock_data['dividends'].to_frame().to_excel(writer, sheet_name='Dividends')
            stock_data['splits'].to_frame().to_excel(writer, sheet_name='Splits')
            pd.DataFrame([stock_data['info']]).to_excel(writer, sheet_name='Info')
    
    def load_all(self, tickers):
        for ticker in tickers:
            self.load_stock(ticker)

# Example usage:
# loader = Loader(source="yfinance")
# loader.load_all(["AAPL", "MSFT", "GOOGL"])
# print(loader.get_stock_info("AAPL"))
# loader.save_stock_data("AAPL", "AAPL_stock_data.xlsx")
