import time
import numpy as np
import logging
from src.common.AssetData import AssetData
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut 
from src.databaseService.OutsourceLoader import OutsourceLoader
from src.common.AssetDataService import AssetDataService

logger = logging.getLogger(__name__)

class EstablishStocks:
    def __init__(self, 
                 tickerList: list[str],
                 operator: str = "yfinance",
                 apiKey: str = ""):
        self.tickerList = tickerList
        self.operator = operator
        self.apiKey = apiKey
    
    def updateAssets(self):
        stockList: list[str] = np.unique(self.tickerList).tolist()
        fileInOut = AssetFileInOut("src/database")
        yamlInOut = YamlTickerInOut("src/stockGroups")
        outsourceLoader = OutsourceLoader(outsourceOperator=self.operator, api_key=self.apiKey)
        allTickersYamlList = []
        
        for ticker in stockList:
            tStart = time.time()
            
            # Load current asset from database
            if fileInOut.exists(ticker):
                asset: AssetData = fileInOut.loadFromFile(ticker)
            else:
                asset: AssetData = AssetDataService.defaultInstance(ticker = ticker)
            
            # Try updating asset data
            try:
                asset_new: AssetData = outsourceLoader.update(asset = asset, ticker = ticker)
                
                fileInOut.saveToFile(asset_new)
                allTickersYamlList.append(ticker)
                
                logger.info(f"Got Stock data for {ticker}.")
                
                # Delay due to max api calls
                tEnd = time.time()
                if self.operator == "alphaVantage":
                    time.sleep(max(8.5 - (tEnd-tStart),0))
                if self.operator == "yfinance":
                    time.sleep(max(2 - (tEnd-tStart),0))
            
            except Exception as e:
                logger.info(f"EXCEPTION. Stock data for {ticker} not retrievable. Error message: {e}")
                
        # Save all tickers to YAML file
        yamlInOut.saveToFile(allTickersYamlList, "group_all")