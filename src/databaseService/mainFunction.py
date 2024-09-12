import yaml # the library is pyyaml not yaml
import os
import time

from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService
from src.common.AssetFileInOut import AssetFileInOut 
from src.databaseService.OutsourceLoader import OutsourceLoader

# Step 1: Load the YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Main function
def mainFunction():
    ## Load tickers
    yamlFile = 'src/databaseService/stockTickers.yaml'
    filePath = os.path.join(os.getcwd(), yamlFile)
    with open(filePath, 'r') as file:
        tickersDict = yaml.safe_load(file)

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