import yaml
import os
import logging
from datetime import datetime

from src.databaseService.EstablishStocks import EstablishStocks
from src.common.YamlTickerInOut import YamlTickerInOut 

from src.stockGroupsService.GroupOver20Years import GroupOver20Years
from src.stockGroupsService.GroupSwiss import GroupSwiss
from src.stockGroupsService.GroupManager import GroupManager
from src.stockGroupsService.GroupSwissOver20Years import GroupSwissOver20Years
from src.stockGroupsService.GroupSnP500 import GroupSnP500
from src.stockGroupsService.GroupSnP500Over20Years import GroupSnP500Over20Years
from src.stockGroupsService.GroupSnP500NAS100Over20Years import GroupSnP500NAS100Over20Years
from src.stockGroupsService.GroupSnP500FinanTo2011 import GroupSnP500FinanTo2011
from src.stockGroupsService.GroupDebug import GroupDebug
from src.stockGroupsService.GroupFinanTo2011 import GroupFinanTo2011
from src.stockGroupsService.GroupFinanTo2016 import GroupFinanTo2016

formatted_date = datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    level=logging.INFO, 
    filename=f"logs/output_updateDatabase_{formatted_date}.log",
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

groupClasses = [
    GroupOver20Years(),
    GroupSwiss(),
    GroupSwissOver20Years(),
    GroupSnP500(),
    GroupSnP500Over20Years(),
    GroupSnP500NAS100Over20Years(),
    GroupSnP500FinanTo2011(),
    GroupDebug(),
    GroupFinanTo2011(),
    GroupFinanTo2016(),
]

def get_apiKey() -> str:
    # Read and load the api key for alpha vantage
    try:
        with open(os.path.join("secrets", "alphaVantage.yaml"), 'r') as file:  # Open the YAML file for reading
            config = yaml.safe_load(file)  # Load the YAML content
            apiKey = config['alphaVantage_premium']['apiKey']  # Access the required key
    except PermissionError:
        logger.error("Permission denied. Please check file permissions.")
    except FileNotFoundError:
        logger.error("File not found. Please verify the path.")
    except KeyError:
        logger.error("KeyError: Check the structure of the YAML file.")
    except yaml.YAMLError as e:
        logger.error("YAML Error:", e)
        
    return apiKey

def get_stock_list() -> list[str]:
    # Read and load the stock list from the YAML file
    tickersDict = YamlTickerInOut("src/tickerSelection").loadFromFile("stockTickers")
    stockList: list = tickersDict[0]['stocks'] #NYSE
    stockList.extend(tickersDict[1]['stocks']) #NASDAQ
    stockList.extend(tickersDict[2]['stocks']) #NYSE MKT
    if operator != "alphaVantage": #Alpha Vantage has no Swiss Data
        for ticker in tickersDict[3]['stocks']:
            if isinstance(ticker, str) and ticker.lower()[0:1] == 'ch':
                stockList.append(ticker)
                continue
            if isinstance(ticker, str) and ticker.lower()[0:1] == 'us':
                continue # discard
            if isinstance(ticker, str) and ticker.lower().endswith('.sw'):
                stockList.append(ticker)
                continue # discard
            stockList.append(ticker + '.SW')
        
    if not stockList:
        raise ValueError("No stocks found in the YAML file.")
        
    return [str(ticker) for ticker in stockList]  # Make sure its a list of strings

if __name__ == "__main__":
    operator = "alphaVantage"  #"yfinance" or "alphaVantage"
    apiKey = get_apiKey() if operator == "alphaVantage" else ""
    stockList = get_stock_list()
    
    
    # Update database entries
    #EstablishStocks(
    #    tickerList=stockList,
    #    operator=operator,
    #    apiKey=apiKey
    #).updateAssets()
    
    
    # Update groups
    dbPath = "src/database"
    groupPath = "src/stockGroups"

    manager = GroupManager(databasePath=dbPath, stockGroupPath=groupPath, groupClasses = groupClasses)
    manager.generateGroups()