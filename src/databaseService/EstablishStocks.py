import time

from src.common.AssetData import AssetData
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut 
from src.databaseService.OutsourceLoader import OutsourceLoader

class EstablishStocks:
    def __init__(self, dirPathManualTicker: str, dirPathLoadedTicker:str, manualYamlName: str, loadedYamlName: str):
        self.dirPathToManualTicker = dirPathManualTicker
        self.dirPathToLoadedTicker = dirPathLoadedTicker
        self.manualYamlName = manualYamlName
        self.loadedYamlName = loadedYamlName

    def stockList(self) -> list:
        # Generate list of all stocks
        tickersDict = YamlTickerInOut(self.dirPathToManualTicker).loadFromFile(self.manualYamlName)

        stockList: list = tickersDict[0]['stocks']
        stockList.extend(tickersDict[1]['stocks'])
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
        stockList.extend(tickersDict[3]['stocks'])
        if not stockList:
            print("No stocks found in the YAML file.")
            return []
        
        return stockList
    
    def loadSaveAssets(self):
        stockList = self.stockList()
        fileOut = AssetFileInOut("src/database")
        outsourceLoader = OutsourceLoader(outsourceOperator="yfinance")
        allTickersYamlList = []
        for ticker in stockList:
            try:
                asset: AssetData = outsourceLoader.load(ticker=ticker)
                fileOut.saveToFile(asset)
                print(f"Got Stock data for {ticker}.")
                allTickersYamlList.append(asset.ticker)
                time.sleep(0.5)
            except:
                pass
        
        YamlTickerInOut(self.dirPathToLoadedTicker).saveToFile(allTickersYamlList, self.loadedYamlName)