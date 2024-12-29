import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService
from src.databaseService.CleanData import CleanData
from src.databaseService.Parser import Parser_AV

class OutsourceLoader:
    outsourceOperator: str  # 'yfinance' or 'alpha_vantage'
    apiKey: str = None     # For alpha_vantage

    def __init__(self, outsourceOperator: str, api_key: str = None):
        self.outsourceOperator = outsourceOperator

        if self.outsourceOperator not in ["yfinance", "alphaVantage"]:
            raise NotImplementedError(f"Operator '{self.outsourceOperator}' not supported.")

        if self.outsourceOperator == "alphaVantage" and api_key is None:
            raise ValueError("API key is required for Alpha Vantage.")

        self.apiKey = api_key
    
    def load(self, ticker: str) -> AssetData:
        assetData: AssetData = AssetDataService.defaultInstance()
        if self.outsourceOperator == "yfinance":
            self._load_from_yfinance(assetData, ticker)
        elif self.outsourceOperator == "alphaVantage":
            self._load_from_alpha_vantage(assetData, ticker)
        
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
            assetData.sector = assetData.about.get('sectorKey','other')
            
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
            CleanData.fill_NAN_to_BusinessDays(assetData.adjClosePrice)
        else:
            raise ValueError("Failed to retrieve Price History for ticker: " + tickerHandle)

        try:
            fullFinancials_qu = stock.quarterly_financials
            fullFinancials_qu = fullFinancials_qu.T
            fullFinancials_qu.index.name = 'Date'
            fullFinancials_qu.index = pd.to_datetime(fullFinancials_qu.index)
            
            fullFinancials_an = stock.financials
            fullFinancials_an = fullFinancials_an.T
            fullFinancials_an.index.name = 'Date'
            fullFinancials_an.index = pd.to_datetime(fullFinancials_an.index)

            assetData.financials_quarterly = fullFinancials_qu.iloc[::-1].reset_index()
            assetData.financials_annually = fullFinancials_an.iloc[::-1].reset_index()
        except:
            warnings.warn("Failed to retrieve Financial Data for ticker: " + tickerHandle)
            
    def _load_from_alpha_vantage(self, assetData: AssetData, tickerHandle: str):
        # Initialize Alpha Vantage API clients
        ts = TimeSeries(key=self.apiKey, output_format='pandas')
        fd = FundamentalData(key=self.apiKey, output_format='pandas')

        try:
            fullSharePrice, _ = ts.get_daily_adjusted(symbol=tickerHandle, outputsize='full')
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            # Log the error or pass as required
            print(f"API call get_daily_adjusted failed for {tickerHandle} error: {str(e)}")
            raise requests.exceptions.RequestException
            
        assetData.ticker = tickerHandle  
        assetData.isin = ""
        
        # Configure time series data
        fullSharePrice = fullSharePrice.iloc[::-1] #flip upside down
        fullSharePrice.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume',
            '7. dividend amount': 'Dividends',
            '8. split coefficient': 'Splits'
        }, inplace=True)
        fullSharePrice.index.name = 'Date'
        #Add utc on the date
        fullSharePrice.index = fullSharePrice.index.tz_localize('UTC')
        assetData.shareprice = fullSharePrice[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        assetData.volume = assetData.shareprice['Volume']
        assetData.adjClosePrice = assetData.shareprice['Adj Close']
        assetData.dividends = fullSharePrice['Dividends']
        assetData.splits = fullSharePrice['Splits']
        CleanData.fill_NAN_to_BusinessDays(assetData.adjClosePrice)
        
        # Configure company overview
        try:
            company_overview, _ = fd.get_company_overview(symbol=tickerHandle)
            
            assetData.about = company_overview.to_dict(orient='records')[0]
            catDict = {
                'OTHER': 'other', 
                'MANUFACTURING':'industrials', 
                'LIFE SCIENCES': 'healthcare', 
                'TECHNOLOGY': 'technology', 
                'FINANCE': 'financial-services', 
                'REAL ESTATE & CONSTRUCTION':'real-estate', 
                'ENERGY & TRANSPORTATION': 'energy', 
                'TRADE & SERVICES': 'consumer-cyclical', 
            }
            assetData.sector = catDict[company_overview["Sector"].iloc[0]]
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            # Log the error or pass as required
            print(f"API call company_overview failed for {tickerHandle}: {str(e)}")
        
        # Configure fundamental data
        try:
            url = 'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol='+tickerHandle+'&apikey='+self.apiKey
            incStatementData = requests.get(url).json()
            url = 'https://www.alphavantage.co/query?function=CASH_FLOW&symbol='+tickerHandle+'&apikey='+self.apiKey
            cashFlowData = requests.get(url).json()
            url = 'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol='+tickerHandle+'&apikey='+self.apiKey
            balanceSheetData = requests.get(url).json()
            url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+tickerHandle+'&apikey='+self.apiKey
            earningsData = requests.get(url).json()
            
            if incStatementData=={} or cashFlowData == {} or balanceSheetData == {} or earningsData == {}:
                raise ImportError(f"Empty Financial Data")
            
            parser = Parser_AV(
                incStatementData=incStatementData, 
                cashFlowData=cashFlowData, 
                balanceSheetData=balanceSheetData, 
                earningsData=earningsData)
            
            assetData.financials_annually, assetData.financials_quarterly = parser.to_pandas()
            
            assetData.financials_quarterly = CleanData.financial_fiscalDateIncongruence(assetData.financials_quarterly)
            assetData.financials_annually = CleanData.financial_fiscalDateIncongruence(assetData.financials_annually)
            
            assetData.financials_annually = CleanData.financial_dropDuplicateYears(assetData.financials_annually)
            assetData.financials_annually = CleanData.financial_dropLastRow(assetData.financials_annually)
            
            #todo: add last row if it is in company overview
            #todo: add  upcoming information from company overview
            
        except (requests.exceptions.RequestException, ValueError, KeyError, ImportError) as e:
            # Log the error or pass as required
            print(f"API call fundametal data failed for {tickerHandle} due to error: {str(e)}")