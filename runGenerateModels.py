from src.predictionModule.CollectionModels import CollectionModels
from src.common.AssetFileInOut import AssetFileInOut

from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
from typing import Dict
import pandas as pd
import numpy as np
import polars as pl
import datetime
import os
import gc

from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

stock_group = "snp500_finanTo2011"
assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_"+stock_group)
assetspl: Dict[str, AssetDataPolars] = {}
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)
    
#To free up RAM
del assets
cutoffDate = pd.Timestamp(year=2011, month=1, day=7, tz='UTC')
for ticker, asset in assetspl.items():
    lastIdx = DPl(assetspl[ticker].adjClosePrice).getNextLowerOrEqualIndex(cutoffDate)
    if not assetspl[ticker].adjClosePrice['Date'].item(lastIdx) == cutoffDate:
        print(f"Cutoff-date {cutoffDate} was not found in ticker {ticker}.")
        
    assetspl[ticker].shareprice = assetspl[ticker].shareprice.slice(lastIdx)
    assetspl[ticker].adjClosePrice = assetspl[ticker].adjClosePrice.slice(lastIdx)
    assetspl[ticker].volume = assetspl[ticker].volume.slice(lastIdx)
    assetspl[ticker].dividends = assetspl[ticker].dividends.slice(lastIdx)
    assetspl[ticker].splits = assetspl[ticker].splits.slice(lastIdx)
    
    assetspl[ticker].shareprice.shrink_to_fit()
    assetspl[ticker].adjClosePrice.shrink_to_fit()
    assetspl[ticker].volume.shrink_to_fit()
    assetspl[ticker].dividends.shrink_to_fit()
    assetspl[ticker].splits.shrink_to_fit()
    
gc.collect()

#assetspl_cutoff: Dict[str, AssetDataPolars] = {} # Testing for leakage
#cutoffDate = pd.Timestamp(year=2024, month=10, day=15, tz='UTC')
#for ticker, asset in assets.items():
    #assetspl_cutoff[ticker] = assetspl[ticker]
    #lastIdx = DPl(assetspl[ticker].adjClosePrice).getNextLowerOrEqualIndex(cutoffDate)
    #if not assetspl[ticker].adjClosePrice['Date'].item(lastIdx) == cutoffDate:
    #    print(f"Cutoff-date {cutoffDate} was not found in ticker {ticker}.")
    #    
    #assetspl_cutoff[ticker].shareprice = assetspl[ticker].shareprice.slice(0, lastIdx + 1)
    #assetspl_cutoff[ticker].adjClosePrice = assetspl[ticker].adjClosePrice.slice(0, lastIdx + 1)
    #assetspl_cutoff[ticker].volume = assetspl[ticker].volume.slice(0, lastIdx + 1)
    #assetspl_cutoff[ticker].dividends = assetspl[ticker].dividends.slice(0, lastIdx + 1)
    #assetspl_cutoff[ticker].splits = assetspl[ticker].splits.slice(0, lastIdx + 1)
    #
    #lastClosePrice = assetspl_cutoff[ticker].shareprice['Close'].last()
    #lastAdjClosePrice = assetspl_cutoff[ticker].adjClosePrice['AdjClose'].last()
    #assetspl_cutoff[ticker].adjClosePrice.with_columns(
    #    (pl.col("AdjClose")*lastClosePrice/lastAdjClosePrice).alias("AdjClose")
    #)
    
startTrainDate=pd.Timestamp(year=2017, month=1, day=4, tz='UTC')
endTrainDate=pd.Timestamp(year=2023, month=11, day=2, tz='UTC')
startValDate=pd.Timestamp(year=2022, month=11, day=1, tz="UTC")
endValDate=pd.Timestamp(year=2023, month=1, day=25, tz="UTC")
startTestDate=pd.Timestamp(year=2023, month=11, day=3, tz='UTC')
endTestDate=pd.Timestamp(year=2024, month=5, day=29, tz="UTC")

spareDatesRatio = 1.0

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
    'idxAfterPrediction': 10,
    'monthsHorizon': 5,
    'timesteps': 5,
    'classificationInterval': [0.05],
    'optuna_trials': 3,
    'LGBM_max_depth': 10,
    'averageOverDays': 5,
    'optuna_weight': 4,
}

if __name__ == "__main__":
    lagList = np.array([0, 10, 20, 30, 45, 55, 69, 80, 110, 150, 240, 280, 320, 366, 420, 600])
    lagList = np.unique(np.random.randint(0, 366*1, 10))
    lagList = np.array([0, 10, 20, 30, 45, 55])
    test_date = pd.Timestamp(year=2024, month=12, day=13, tz='UTC')
    res = []
    for dayLag in lagList:
        test_date_lag = test_date - pd.Timedelta(days=dayLag)
        
        formatted_date = test_date_lag.strftime('%d%b%Y')
        
        subsetML_binaries_subsetml_name = (
            f"SubsetML_{stock_group}_spareDate{int(100 * spareDatesRatio)}_{formatted_date}_10days"
        )
        akinML_binaries_subsetml_name = (
            f"AkinDistriML_{stock_group}_{formatted_date}_10days"
        )
        
        print(f"----------Date: {test_date_lag} , Lag: {dayLag}----------")
        print(f"----------{akinML_binaries_subsetml_name}----------")
        filePath = os.path.join("src/predictionModule/bin", akinML_binaries_subsetml_name + '.pkl')
        if not os.path.exists(filePath):
            CollectionModels.AkinDistriML_saveData(
                assetspl = assetspl, 
                save_name = akinML_binaries_subsetml_name,
                params = params,
                test_date = test_date_lag) 
            
        res_loc = CollectionModels.AkinDistriML_loadup_analyze(
            assetspl = assetspl, 
            loadup_name = akinML_binaries_subsetml_name,
            test_date = test_date_lag,
            params = params,)
        res.append(res_loc)
         
        
        #print(f"----------{subsetML_binaries_subsetml_name}----------")
        #filePath = os.path.join("src/predictionModule/bin", subsetML_binaries_subsetml_name + '.pkl')
        #if not os.path.exists(filePath):
        #    CollectionModels.SubsetML_saveData(
        #        assetspl = assetspl, 
        #        save_name = subsetML_binaries_subsetml_name,
        #        spareRatio=spareDatesRatio,
        #        params = params,
        #        test_date = test_date_lag)  
        #
        #CollectionModels.SubsetML_loadup_analyze(
        #    assetspl = assetspl, 
        #    loadup_name = subsetML_binaries_subsetml_name,
        #    test_date = test_date_lag,
        #    params = params,)
        #
        #CollectionModels.SubsetML_loadup_predict(
        #    assetspl = assetspl, 
        #    loadup_name = binaries_subsetml_name,
        #    test_date = test_date_lag,
        #    params=params,)
    
    print(f"Resulting list: {res}")
    print(f"Resulting mean: {np.mean([x for x in res if x is not None])}")