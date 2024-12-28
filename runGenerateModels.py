from src.predictionModule.CollectionModels import CollectionModels
from src.common.AssetFileInOut import AssetFileInOut

from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
from typing import Dict
import pandas as pd

assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_finanTo2011")
assetspl: Dict[str, AssetDataPolars] = {}
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)
    
startTrainDate=pd.Timestamp(year=2015, month=1, day=4, tz='UTC')
endTrainDate=pd.Timestamp(year=2018, month=3, day=4, tz='UTC')
startValDate=pd.Timestamp(year=2017, month=3, day=4, tz="UTC")
endValDate=pd.Timestamp(year=2017, month=3, day=14, tz="UTC")
startTestDate=pd.Timestamp(year=2018, month=3, day=5, tz='UTC')
endTestDate=pd.Timestamp(year=2018, month=3, day=11, tz="UTC")

spareDatesRatio = 0.7

spare_dates_train = CollectionModels.sample_spare_dates(startTrainDate, endTrainDate, spareDatesRatio)
spare_dates_val1 = CollectionModels.sample_spare_dates(startValDate, endValDate, spareDatesRatio)
spare_dates_val2 = CollectionModels.sample_spare_dates(startValDate - pd.Timedelta(days=365), endValDate - pd.Timedelta(days=365),  spareDatesRatio)
spare_dates_val3 = CollectionModels.sample_spare_dates(startValDate - pd.Timedelta(days=365), endValDate - pd.Timedelta(days=365),  spareDatesRatio)
spare_dates_test = CollectionModels.sample_spare_dates(startTestDate, endTestDate, spareDatesRatio)

spare_dates_val = spare_dates_val1.union(spare_dates_val2).union(spare_dates_val3)

params = {
    'daysAfterPrediction': 21*6,
    'monthsHorizon': 6,
    'timesteps': 5,
    'classificationInterval': [0], 
    'optuna_trials': 2,
    'LGBM_max_depth': 10
}

if __name__ == "__main__":
    binaries_name = "NextDayML_debug_test2015_nexthalfyear"
    
    #CollectionModels.NextDayML_saveData(
    #    assetspl = assetspl, 
    #    save_name = binaries_name, 
    #    trainDates = spare_dates_train, 
    #    valDates = spare_dates_val, 
    #    testDates = spare_dates_test,
    #    params=params)
    CollectionModels.NextDayML_loadupData_lgbm(
        assetspl=assetspl, 
        loadup_name = binaries_name, 
        params=params)