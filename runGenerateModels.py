from src.predictionModule.CollectionModels import CollectionModels
from src.common.AssetFileInOut import AssetFileInOut

from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
from typing import Dict
import pandas as pd
import numpy as np
import datetime

from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_finanTo2011")
assetspl: Dict[str, AssetDataPolars] = {}
assetspl_cutoff: Dict[str, AssetDataPolars] = {} # Testing for leakage
cutoffDate = pd.Timestamp(year=2024, month=10, day=15, tz='UTC')
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)
    assetspl_cutoff[ticker] = assetspl[ticker]
    lastIdx = DPl(assetspl[ticker].adjClosePrice).getNextLowerOrEqualIndex(cutoffDate)
    if not assetspl[ticker].adjClosePrice['Date'].item(lastIdx) == cutoffDate:
        print(f"Cutoff-date {cutoffDate} was not found in ticker {ticker}.")
        
    assetspl_cutoff[ticker].shareprice = assetspl[ticker].shareprice.slice(0, lastIdx + 1)
    assetspl_cutoff[ticker].adjClosePrice = assetspl[ticker].adjClosePrice.slice(0, lastIdx + 1)
    assetspl_cutoff[ticker].volume = assetspl[ticker].volume.slice(0, lastIdx + 1)
    assetspl_cutoff[ticker].dividends = assetspl[ticker].dividends.slice(0, lastIdx + 1)
    assetspl_cutoff[ticker].splits = assetspl[ticker].splits.slice(0, lastIdx + 1)
    
startTrainDate=pd.Timestamp(year=2017, month=1, day=4, tz='UTC')
endTrainDate=pd.Timestamp(year=2023, month=11, day=2, tz='UTC')
startValDate=pd.Timestamp(year=2022, month=11, day=1, tz="UTC")
endValDate=pd.Timestamp(year=2022, month=11, day=25, tz="UTC")
startTestDate=pd.Timestamp(year=2023, month=11, day=3, tz='UTC')
endTestDate=pd.Timestamp(year=2023, month=11, day=29, tz="UTC")

spareDatesRatio = 0.75

spare_dates_train = CollectionModels.sample_spare_dates(startTrainDate, endTrainDate, spareDatesRatio)
spare_dates_val1 = CollectionModels.sample_spare_dates(startValDate, endValDate, 1.0)
spare_dates_val2 = CollectionModels.sample_spare_dates(startValDate - pd.Timedelta(days=365), endValDate - pd.Timedelta(days=365),  1.0)
spare_dates_val3 = CollectionModels.sample_spare_dates(startValDate - pd.Timedelta(days=365*2), endValDate - pd.Timedelta(days=365*2),  1.0)
spare_dates_val4 = CollectionModels.sample_spare_dates(startValDate - pd.Timedelta(days=365*3), endValDate - pd.Timedelta(days=365*3),  1.0)
spare_dates_val5 = CollectionModels.sample_spare_dates(startValDate - pd.Timedelta(days=365*4), endValDate - pd.Timedelta(days=365*4),  1.0)
spare_dates_val6 = CollectionModels.sample_spare_dates(startValDate - pd.Timedelta(days=365*5), endValDate - pd.Timedelta(days=365*5),  1.0)

spare_dates_test = CollectionModels.sample_spare_dates(startTestDate, endTestDate, 1.0)

spare_dates_val = spare_dates_val1.union(spare_dates_val2).union(spare_dates_val3).union(spare_dates_val4).union(spare_dates_val5).union(spare_dates_val6)

spare_dates_train = spare_dates_train.difference(spare_dates_val)
#spare_dates_val_half = np.random.choice(spare_dates_val, size=max(1,len(spare_dates_val) // 2), replace=False)
#spare_dates_val = spare_dates_val.difference(spare_dates_val_half)
#spare_dates_train = spare_dates_train_cleaned.union(pd.DatetimeIndex(spare_dates_val_half))

assert spare_dates_train.intersection(spare_dates_val).empty
params = {
    'daysAfterPrediction': 21,
    'monthsHorizon': 6,
    'timesteps': 5,
    'classificationInterval': [0, 0.05],
    'optuna_trials': 20,
    'optuna_trials': 20,
    'LGBM_max_depth': 20,
    'averageOverDays': 5,
}

if __name__ == "__main__":
    #formatted_date = datetime.datetime.now().strftime('%d%m%y')
    binaries_name = f"ConditionalML_debug_oneMonthAfter_Over5perc_LeakageCheck"
    
    #CollectionModels.ConditionalML_saveData(
    #    assetspl = assetspl_cutoff, 
    #    save_name = binaries_name, 
    #    trainDates = spare_dates_train, 
    #    valDates = spare_dates_val, 
    #    testDates = spare_dates_test,
    #    params=params)
    CollectionModels.CondtionalML_loadupData_lgbm(
        assetspl=assetspl_cutoff, 
        loadup_name = binaries_name, 
        params=params)