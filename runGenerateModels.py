from src.predictionModule.CollectionModels import CollectionModels
from src.common.AssetFileInOut import AssetFileInOut

from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
from typing import Dict
import pandas as pd

assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_finanTo2011")
#assets = {}
#tickers = ['GOOGL', 'AAPL', 'MSFT']
#for ticker in tickers:
#    assets[ticker] = AssetFileInOut("src/database").loadFromFile(ticker)
# Convert to Polars for speedup
assetspl: Dict[str, AssetDataPolars] = {}
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)
    
startTrainDate=pd.Timestamp(year=2015, month=9, day=4, tz='UTC')
endTrainDate=pd.Timestamp(year=2018, month=2, day=4, tz='UTC')
startValDate=pd.Timestamp(year=2017, month=2, day=3, tz="UTC")
endValDate=pd.Timestamp(year=2017, month=2, day=13, tz="UTC")
startTestDate=pd.Timestamp(year=2018, month=2, day=5, tz='UTC')
endTestDate=pd.Timestamp(year=2018, month=2, day=11, tz="UTC")

spareDatesRatio = 0.5

spare_dates_train = CollectionModels.sample_spare_dates(startTrainDate, endTrainDate, spareDatesRatio)
spare_dates_val   = CollectionModels.sample_spare_dates(startValDate,   endValDate,   spareDatesRatio)
spare_dates_test  = CollectionModels.sample_spare_dates(startTestDate,  endTestDate,  spareDatesRatio)

if __name__ == "__main__":
    binaries_name = "NextDayML_debug_test2015_nextyear"
    
    CollectionModels.NextDayML_saveData(assetspl = assetspl, save_name = binaries_name, trainDates = spare_dates_train, valDates = spare_dates_val, testDates = spare_dates_test)
    CollectionModels.NextDayML_loadupData_lgbm(assetspl=assetspl, loadup_name = binaries_name)