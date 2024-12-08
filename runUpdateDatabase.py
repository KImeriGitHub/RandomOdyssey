import yaml
import os
from src.databaseService.EstablishStocks import EstablishStocks

def get_apiKey() -> str:
    # Read and load the api key for alpha vantage
    try:
        with open(os.path.join("secrets", "alphaVantage.yaml"), 'r') as file:  # Open the YAML file for reading
            config = yaml.safe_load(file)  # Load the YAML content
            apiKey = config['alphaVantage_premium']['apiKey']  # Access the required key
    except PermissionError:
        print("Permission denied. Please check file permissions.")
    except FileNotFoundError:
        print("File not found. Please verify the path.")
    except KeyError:
        print("KeyError: Check the structure of the YAML file.")
    except yaml.YAMLError as e:
        print("YAML Error:", e)
        
    return apiKey

if __name__ == "__main__":
    operator = "alphaVantage"  #"yfinance" or "alphaVantage"
    
    apiKey = get_apiKey() if operator == "alphaVantage" else ""
    
    EstablishStocks(dirPathManualTicker = "src/tickerSelection",
               dirPathLoadedTicker = "src/stockGroups",
               manualYamlName = "stockTickers",
               loadedYamlName = "group_all",
               operator=operator,
               apiKey=apiKey).loadSaveAssets()
