from databaseService.LoadStocks import EstablishStocks

if __name__ == "__main__":
    EstablishStocks(dirPathManualTicker = "src/tickerSelection",
               dirPathLoadedTicker = "src/stockGroups",
               manualYamlName = "stockTickers",
               loadedYamlName = "group_all").loadSaveAssets()
