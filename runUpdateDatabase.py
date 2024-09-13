from databaseService.LoadStocks import LoadStocks

if __name__ == "__main__":
    LoadStocks(dirPathManualTicker = "src/tickerSelection",
               dirPathLoadedTicker = "src/stockGroups",
               manualYamlName = "stockTickers",
               loadedYamlName = "group_all").loadSaveAssets()
