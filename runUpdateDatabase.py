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
from src.stockGroupsService.GroupRegOHLCVOver5Years import GroupOHLCVOver5Years

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
    GroupOHLCVOver5Years()
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
    tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("stockTickers")

    # Build a lookup by stockExchange name (case-insensitive)
    sections = {
        str(entry.get("stockExchange", "")).strip().upper(): entry.get("stocks", [])
        for entry in tickers
        if isinstance(entry, dict)
    }

    stockList: list[str] = []
    stockList += sections.get("NYSE", [])
    stockList += sections.get("NASDAQ", [])
    # NYSE MKT may also appear as NYSEMKT or AMEX; take the first non-empty
    stockList += (sections.get("NYSE MKT") or sections.get("NYSEMKT") or sections.get("AMEX") or [])
    stockList += (sections.get("NYSE ARCA") or sections.get("ARCA") or [])
    stockList += (sections.get("BATS") or [])

    # Handle Swiss tickers unless using Alpha Vantage
    if operator != "alphaVantage":  # Alpha Vantage has no Swiss Data
        swiss = (sections.get("SIX") or sections.get("SWISS") or [])
        for ticker in swiss:
            if not isinstance(ticker, str):
                continue

            t = ticker.strip()
            tl = t.lower()
            if tl.startswith("ch"):
                stockList.append(t)
            elif tl.startswith("us"):
                continue  # discard
            elif tl.endswith(".sw"):
                stockList.append(t)
            else:
                stockList.append(t + ".SW")
        
    if not stockList:
        raise ValueError("No stocks found in the YAML file.")
        
    # Make sure its a list of strings
    return [str(ticker) for ticker in stockList]

if __name__ == "__main__":
    operator = "alphaVantage"  #"yfinance" or "alphaVantage"
    apiKey = get_apiKey() if operator == "alphaVantage" else ""
    stockList = get_stock_list()
    
    es = EstablishStocks(
        tickerList=stockList,
        operator=operator,
        apiKey=apiKey
    )
    
    dbPath = "src/database"
    groupPath = "src/stockGroups"
    gm = GroupManager(databasePath=dbPath, stockGroupPath=groupPath, groupClasses = groupClasses)
    
    # Update database entries
    es.updateAssets()
    
    # Update groups
    gm.generateGroups()
    
    # Validate datbase entries
    es.validateAssets()
    