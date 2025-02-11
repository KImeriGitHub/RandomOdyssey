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
import logging

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

params = {
    'idxAfterPrediction': 10,
    'monthsHorizon': 5,
    'timesteps': 5,
    'classificationInterval': [0.05],
    
    'optuna_trials': 3,
    'averageOverDays': 5,
    'optuna_weight': 4,
    
    'Akin_test_quantile': 0.7,
    'Akin_feature_max': 800,
    'Akin_itersteps': 6,
    'Akin_pre_num_leaves': 1024,
    'Akin_pre_num_boost_round': 1000,
    'Akin_pre_weight_truncation': 8,
    'Akin_num_leaves': 1024,
    'Akin_num_boost_round': 1000,
    'Akin_top_highest': 10,
}

#if __name__ == "__main__":
#    eval_date = pd.Timestamp(year=2025, month=2, day=7, tz='UTC')
#    formatted_date = eval_date.strftime('%d%b%Y')
#    akinML_binaries_live_name = (
#            f"AkinDistriML_{stock_group}_{formatted_date}_10days"
#        )
#    
#    print(f"----------Date: {eval_date}----------")
#    print(f"----------{akinML_binaries_live_name}----------")
#    
#    logging.basicConfig(
#        filename='output.txt',
#        level=logging.DEBUG,
#        format='%(asctime)s - %(levelname)s - %(message)s'
#    )
#    logger = logging.getLogger(__name__)
#
#    CollectionModels.AkinDistriML_loadUpData_predict(
#        assetspl = assetspl, 
#        loadup_name = akinML_binaries_live_name,
#        params = params,
#        test_date = eval_date
#    ) 
    
if __name__ == "__main__":
    lagList = np.array([0, 10, 20, 30, 45, 55, 69, 80, 110, 150, 240, 280, 320, 366, 420, 600])
    lagList = np.unique(np.random.randint(365*0, 366*1, 4))
    lagList = np.array([10])
    test_date = pd.Timestamp(year=2024, month=12, day=13, tz='UTC')
    res = []
    for dayLag in lagList:
        test_date_lag = test_date - pd.Timedelta(days=dayLag)
        
        formatted_date = test_date_lag.strftime('%d%b%Y')
        
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
    
    print(f"Resulting list: {res}")
    print(f"Resulting mean: {np.mean([x for x in res if x is not None])}")