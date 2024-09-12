import time

from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlInOut import YamlInOut 
from src.databaseService.OutsourceLoader import OutsourceLoader

def mainFunction():
    ## Load tickers
    yamlInOut = YamlInOut('src/databaseService')
    tickersDict = yamlInOut.loadFromFile("stockTickers")

    stockList: list = tickersDict[0]['stocks']
    stockList.extend(tickersDict[1]['stocks'])
    [stockList.append(a+".SW") for a in tickersDict[2]['stocks']]
    if not stockList:
        print("No stocks found in the YAML file.")
        return

    ## Save stock data
    fileOut = AssetFileInOut("src/database")
    outsourceLoader = OutsourceLoader(outsourceOperator="yfinance")
    for ticker in stockList:
        try:
            asset: AssetData = outsourceLoader.load(ticker=ticker)
            fileOut.saveToFile(asset)
            print(f"Got Stock data for {ticker}.")
            time.sleep(1)
        except:
            pass