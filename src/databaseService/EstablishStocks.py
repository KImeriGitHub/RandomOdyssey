import time

from src.common.AssetData import AssetData
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut 
from src.databaseService.OutsourceLoader import OutsourceLoader

class EstablishStocks:
    def __init__(self, 
                 dirPathManualTicker: str, 
                 dirPathLoadedTicker:str, 
                 manualYamlName: str, 
                 loadedYamlName: str,
                 operator: str = "yfinance",
                 apiKey: str = ""):
        self.dirPathToManualTicker = dirPathManualTicker
        self.dirPathToLoadedTicker = dirPathLoadedTicker
        self.manualYamlName = manualYamlName
        self.loadedYamlName = loadedYamlName
        self.operator = operator
        self.apiKey = apiKey

    def __stockList(self, operator: str = "") -> list:
        # Generate list of all stocks
        tickersDict = YamlTickerInOut(self.dirPathToManualTicker).loadFromFile(self.manualYamlName)

        stockList: list = tickersDict[0]['stocks']
        stockList.extend(tickersDict[1]['stocks'])
        stockList.extend(tickersDict[3]['stocks'])
        if operator != "alphaVantage": #Alpha Vantage has no Swiss Data
            for ticker in tickersDict[2]['stocks']:
                if isinstance(ticker, str) and ticker.lower()[0:1] == 'ch':
                    stockList.append(ticker)
                    continue
                if isinstance(ticker, str) and ticker.lower()[0:1] == 'us':
                    continue # discard
                if isinstance(ticker, str) and ticker.lower().endswith('.sw'):
                    stockList.append(ticker)
                    continue # discard
                stockList.append(ticker+'.SW')
            
        if not stockList:
            raise ValueError("No stocks found in the YAML file.")
        
        return stockList
    
    def loadSaveAssets(self):
        stockList = self.__stockList()
        fileOut = AssetFileInOut("src/database")
        outsourceLoader = OutsourceLoader(outsourceOperator=self.operator, api_key=self.apiKey)
        allTickersYamlList = []
        for ticker in stockList:
            try:
                tStart = time.time()
                asset: AssetData = outsourceLoader.load(ticker=ticker)
                fileOut.saveToFile(asset)
                print(f"Got Stock data for {ticker}.")
                allTickersYamlList.append(asset.ticker)
                tEnd = time.time()
                if self.operator == "alphaVantage":
                    time.sleep(max(8.5 - (tEnd-tStart),0)) # due to max api calls
                if self.operator == "yfinance":
                    time.sleep(max(2 - (tEnd-tStart),0))
            except:
                print(f"EXCEPTION. Stock data for {ticker} not retrievable.")
        
        YamlTickerInOut(self.dirPathToLoadedTicker).saveToFile(allTickersYamlList, self.loadedYamlName)