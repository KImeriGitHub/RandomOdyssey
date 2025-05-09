import yfinance as yf
import pandas as pd
import numpy as np
import logging
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import requests

from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService
from src.databaseService.CleanData import CleanData
from src.databaseService.Parser import Parser_AV
from src.databaseService.Merger import Merger_AV

logger = logging.getLogger(__name__)

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
    
    def update(self, asset: AssetData, ticker: str) -> AssetData:
        new_asset: AssetData
        if self.outsourceOperator == "yfinance":
            raise NotImplementedError("yfinance is not implemented yet.")
        elif self.outsourceOperator == "alphaVantage":
            new_asset = self._from_alpha_vantage(asset, ticker)
        
        return new_asset
            
    def _from_alpha_vantage(self, assetData: AssetData, tickerHandle: str):
        # Initialize Alpha Vantage API clients
        ts = TimeSeries(key=self.apiKey, output_format='pandas')
        fd = FundamentalData(key=self.apiKey, output_format='pandas')
        
        ## Check if tickerHandle returns daily. If not, raise error.
        try:
            fullSharePrice, _ = ts.get_daily_adjusted(symbol=tickerHandle, outputsize='full')
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            logger.info(f"API call get_daily_adjusted failed for {tickerHandle} error: {str(e)}")
            raise requests.exceptions.RequestException
            
        new_asset = AssetDataService.defaultInstance(ticker=tickerHandle)
        new_asset = AssetDataService.copy(ad=assetData)
        
        mergerService = Merger_AV(assetData=new_asset)
        
        ## Merge daily shareprices
        mergerService.merge_shareprice(mergingshareprice=fullSharePrice)
        
        ## Configure company overview
        company_overview = None
        try:
            company_overview, _ = fd.get_company_overview(symbol=tickerHandle)
            
            new_asset.about = company_overview.to_dict(orient='records')[0]
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
            new_asset.sector = catDict[company_overview["Sector"].iloc[0]]
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            # Log the error or pass as required
            logger.info(f"API call company_overview failed for {tickerHandle}: {str(e)}")
        
        ## Configure fundamental data
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

            financials_annually, financials_quarterly = parser.to_pandas()
            
            financials_quarterly = CleanData.financial_fiscalDateIncongruence(financials_quarterly, daysDiscrep = 15)
            financials_annually = CleanData.financial_fiscalDateIncongruence(financials_annually, daysDiscrep = 60)
            financials_annually = CleanData.financial_lastRow_removeIfOutOfFiscal(financials_annually)
            
            mergerService.merge_financials(financials_annually, financials_quarterly)
            
        except (requests.exceptions.RequestException, ValueError, KeyError, ImportError) as e:
            # Log the error or pass as required
            logger.info(f"API call fundametal data failed for {tickerHandle} due to error: {str(e)}")
            
        return new_asset