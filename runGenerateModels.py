from src.predictionModule.CollectionModels import CollectionModels
from src.common.AssetFileInOut import AssetFileInOut

from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
from typing import Dict

assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_finanTo2011")
#assets = {}
#tickers = ['GOOGL', 'AAPL', 'MSFT']
#for ticker in tickers:
#    assets[ticker] = AssetFileInOut("src/database").loadFromFile(ticker)
# Convert to Polars for speedup
assetspl: Dict[str, AssetDataPolars] = {}
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)

if __name__ == "__main__":
    binaries_name = "NextDayML_debug_test2015"
    #CollectionModels.fourierML_saveData(assetspl=assetspl, save_name = binaries_name)
    #CollectionModels.fourierML_loadupData_xgb(assetspl=assetspl, loadup_name = binaries_name)
    #CollectionModels.fourierML_loadupData_rp(assetspl=assetspl, loadup_name = binaries_name)
    #CollectionModels.fourierML_loadupData_LSTM(assetspl=assetspl, loadup_name = binaries_name)
    
    CollectionModels.NextDayML_saveData(assetspl = assetspl, save_name = binaries_name)
    #CollectionModels.NextDayML_loadupData_xgb(assetspl=assetspl, loadup_name = binaries_name)
    CollectionModels.NextDayML_loadupData_lgbm(assetspl=assetspl, loadup_name = binaries_name)
    #CollectionModels.NextDayML_loadupData_LSTM(assetspl=assetspl, loadup_name = binaries_name)