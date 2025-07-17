import time
import numpy as np
import requests

from src.common.AssetData import AssetData
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut 
from src.databaseService.OutsourceLoader import OutsourceLoader
from src.common.AssetDataService import AssetDataService
from src.databaseService.Merger import Merger
from src.databaseService.Parser import Parser_AV

import logging
logger = logging.getLogger(__name__)

class EstablishStocks:
    def __init__(self, 
                 tickerList: list[str],
                 operator: str = "yfinance",
                 apiKey: str = ""):
        self.tickerList = tickerList
        self.operator = operator
        self.apiKey = apiKey
        
        self.fileInOut = AssetFileInOut("src/database")
        self.yamlInOut = YamlTickerInOut("src/stockGroups")
    
    def updateAssets(self):
        """
        Load up the assets from the database using the stocklist supplied by tickerList. If not given, then use default instance.
        Then update the asset data, such that if the shareprice cannot be loaded up: abort. Otherwise try getting financials.
        """
        stockList: list[str] = np.unique(self.tickerList).tolist()
        outsourceLoader = OutsourceLoader(outsourceOperator=self.operator, api_key=self.apiKey)
        allTickersYamlList = []
        
        for ticker in stockList:
            tStart = time.time()
            
            # Load current asset from database
            if self.fileInOut.exists(ticker):
                asset: AssetData = self.fileInOut.loadFromFile(ticker)
            else:
                asset: AssetData = AssetDataService.defaultInstance(ticker = ticker)
            new_asset: AssetData = AssetDataService.copy(asset)
            mergerService = Merger(assetData=new_asset)
            
            # Try getting shareprice data
            try:
                loadad_shareprice = outsourceLoader.request_shareprice(ticker = ticker)  # Here using alpha_vantage.timeseries as mediary
                loadad_shareprice = Parser_AV(sharepriceData=loadad_shareprice).parse_shareprice()
                
                mergerService.merge_shareprice(mergingshareprice=loadad_shareprice)
                
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                # Handle exceptions
                logger.info(f"EXCEPTION. Shareprice data for {ticker} not retrievable.")
                continue
            
            # Try getting about and sector
            try:
                loadad_overview = outsourceLoader.request_company_overview(ticker = ticker)
                loadad_about, loaded_sector = Parser_AV(overview=loadad_overview).parse_overview()
                
                mergerService.asset.about = loadad_about
                mergerService.asset.sector = loaded_sector
                
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                pass
            
            #Try getting financials
            try:
                incStatementData, cashFlowData, balanceSheetData, earningsData = outsourceLoader.request_financials(ticker = ticker)
                financials_annually, financials_quarterly = Parser_AV(
                    incStatementData = incStatementData, 
                    cashFlowData = cashFlowData, 
                    balanceSheetData = balanceSheetData, 
                    earningsData = earningsData
                ).parse_financials()
                
                mergerService.merge_financials(fin_ann=financials_annually, fin_quar=financials_quarterly)
                
            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                pass
            
            # Save asset and append ticker to group_all list
            self.fileInOut.saveToFile(new_asset)
            allTickersYamlList.append(ticker)
            
            logger.info(f"Got Stock data for {ticker}.")
            
            # Delay due to max api calls
            tEnd = time.time()
            if self.operator == "alphaVantage":
                time.sleep(max(8.5 - (tEnd-tStart),0))
            if self.operator == "yfinance":
                time.sleep(max(2 - (tEnd-tStart),0))
                
        # Save all tickers to YAML file
        self.yamlInOut.saveToFile(allTickersYamlList, "group_all")
        
    def validateAssets(self):
        """
        Validate the assets in the group_all list.
        """
        yamlList = self.yamlInOut.loadFromFile("group_all")
        
        for ticker in yamlList:
            logger.info(f"Validating {ticker}...")
            # Load current asset from database
            if self.fileInOut.exists(ticker):
                asset: AssetData = self.fileInOut.loadFromFile(ticker)
            else:
                logger.error(f"VALIDATION ERROR: Asset {ticker} not found in database.")
                continue
                
            if not isinstance(asset, AssetData):
                logger.error(f"VALIDATION ERROR: Asset {ticker} is not of type AssetData.")
                continue
            
            # Check if shareprice DataFrame is empty
            if asset.shareprice.empty:
                logger.error(f"VALIDATION ERROR: Asset {ticker} has an empty shareprice DataFrame.")
            else:
                AssetDataService.validate_asset_data(asset)
                
            