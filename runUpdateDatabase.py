import yaml
import os
import logging
from datetime import datetime
from src.databaseService.EstablishStocks import EstablishStocks
from src.common.YamlTickerInOut import YamlTickerInOut 

formatted_date = datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    level=logging.INFO, 
    filename=f"output_updateDatabase_{formatted_date}.log",
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    EstablishStocks(
        tickerList=stockList,
        operator=operator,
        apiKey=apiKey
    ).updateAssets()
