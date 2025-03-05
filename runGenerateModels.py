from src.predictionModule.CollectionModels import CollectionModels
from src.common.AssetFileInOut import AssetFileInOut

from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
from typing import Dict
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime
import os
import gc
import logging
import optuna

from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

stock_group = "snp500_finanTo2011"
assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_"+stock_group)
assetspl: Dict[str, AssetDataPolars] = {}
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)

#To free up RAM
del assets
cutoffDate = pd.Timestamp(year=2010, month=1, day=7, tz='UTC')
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
    'monthsHorizon': 3,
    'timesteps': 5,
    'classificationInterval': [0.05],
    
    'target_option': 'last',
    
    'optuna_trials': 3,
    'averageOverDays': 5,
    'optuna_weight': 4,
    
    'Akin_test_quantile': 1.0,
    'Akin_feature_max': 1500,
    
    'Akin_itersteps': 3,
    'Akin_pre_weight_truncation': 2,
    'Akin_pre_num_leaves': 1200,
    'Akin_pre_num_boost_round': 1000,
    'Akin_num_leaves': 1200,
    'Akin_num_boost_round': 2000,
    
    'Akin_feature_fraction': 0.95,
    'Akin_top_highest': 10,
    
    'Akin_max_depth': 15, #default = -1
    'Akin_learning_rate': 0.02, #default = 0.1
    'Akin_min_data_in_leaf': 60, # default = 20
    'Akin_min_gain_to_split': 0.02, # default = 0
    'Akin_path_smooth': 0.1, # default = 0
    'Akin_min_sum_hessian_in_leaf': 0.005, #default = 1e-3
}
formatted_date = datetime.now().strftime("%d%b%y_%H%M").lower()

logging.basicConfig(
    filename=f'output_{stock_group}_Akin_{formatted_date}.txt',
    level=logging.DEBUG,
    format='%(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f" Params: {params}")

#if __name__ == "__main__":
#    eval_date = pd.Timestamp(year=2025, month=2, day=28, tz='UTC')
#    formatted_date = eval_date.strftime('%d%b%Y')
#    akinML_binaries_live_name = (
#        f"AkinDistriML_{stock_group}_{formatted_date}_10days"
#    )
#    
#    print(f"----------Date: {eval_date}----------")
#    print(f"----------{akinML_binaries_live_name}----------")
#
#    CollectionModels.AkinDistriML_saveData_predict(
#        assetspl = assetspl, 
#        save_name = akinML_binaries_live_name,
#        params = params,
#        test_date = eval_date,
#        logger = logger,
#    ) 
#    #CollectionModels.AkinDistriML_loadUpData_predict(
#    #    assetspl = assetspl, 
#    #    loadup_name = akinML_binaries_live_name,
#    #    params = params,
#    #    test_date = eval_date,
#    #    logger = logger,
#    #) 

if __name__ == "__main__":
    lagList = np.array([0, 10, 20, 30, 45, 55, 69, 80, 110, 150, 240, 280, 320, 366, 420, 600])
    lagList = np.unique(np.random.randint(366*6+150, 366*7+150, 13))
    
    lagList = np.array([10, 22,  36, 39, 49, 89, 112, 117, 128,
     132,201, 209, 203,263, 268, 288, 325, 384, 414, 489, 594, 
     606, 608, 616, 637, 720, 725, 772, 778, 783, 805, 810, 866, 
     873, 901, 1013, 1037, 1043, 1060, 1064, 1098, 1153, 1172, 
     1242, 1254, 1331, 1337, 1341, 1360, 1404, 1421, 1452, 1487, 
     1509, 1546, 1556, 1557, 1598, 1706, 1707, 1728, 1734, 1775, 
     1802, 1817, 2387, 2411, 2461, 2471, 2500, 2505, 2517, 2536])
    eval_date = pd.Timestamp(year=2024, month=12, day=13, tz='UTC')
    res = []
    for dayLag in lagList:
        test_date_lag = eval_date - pd.Timedelta(days=dayLag)
        
        formatted_date = test_date_lag.strftime('%d%b%Y')
        
        akinML_binaries_subsetml_name = (
            f"AkinDistriML_{stock_group}_{formatted_date}_10days"
        )
        
        logger.info(f"----------Date: {test_date_lag} , Lag: {dayLag}----------")
        logger.info(f"----------{akinML_binaries_subsetml_name}----------")
        filePath = os.path.join("src/predictionModule/bin", akinML_binaries_subsetml_name + '.pkl')
        if not os.path.exists(filePath):
            CollectionModels.AkinDistriML_saveData(
                assetspl = assetspl, 
                save_name = akinML_binaries_subsetml_name,
                params = params,
                test_date = test_date_lag,
                logger = logger,
            ) 
        
        res_loc = CollectionModels.AkinDistriML_loadup_analyze(
            assetspl = assetspl, 
            loadup_name = akinML_binaries_subsetml_name,
            test_date = test_date_lag,
            params = params,
            logger = logger,
        )
        res.append(res_loc)
        
    logger.info(f"Resulting list: {res}")
    logger.info(f"Resulting mean: {np.mean([x for x in res if x is not None])}")

#if __name__ == "__main__":
#    eval_date = pd.Timestamp(year=2024, month=12, day=13, tz='UTC')
#    res = []      
#    test_date_lag = eval_date - pd.Timedelta(days=1224)
#    formatted_date = test_date_lag.strftime('%d%b%Y')
#        
#    akinML_binaries_subsetml_name = (
#        f"AkinDistriML_{stock_group}_{formatted_date}_10days"
#    )
#    def objective(trial):
#        # Create a copy of the base params and override the ones to be optimized
#        params_opt = params.copy()
#        params_opt['Akin_feature_max'] = trial.suggest_int('Akin_feature_max', 300, 2000)
#        params_opt['Akin_pre_num_leaves'] = trial.suggest_int('Akin_pre_num_leaves', 500, 1000)
#        params_opt['Akin_num_leaves'] = trial.suggest_int('Akin_num_leaves', 500, 1000)
#        params_opt['Akin_pre_weight_truncation'] = trial.suggest_int('Akin_pre_weight_truncation', 1, 4)
#        params_opt['Akin_feature_fraction'] = trial.suggest_float('Akin_feature_fraction', 0.01, 0.9)
#        params_opt['Akin_max_depth'] = trial.suggest_int('Akin_max_depth', 5, 12)
#        params_opt['Akin_learning_rate'] = trial.suggest_float('Akin_learning_rate', 0.005, 0.3, log=True)
#        params_opt['Akin_min_data_in_leaf'] = trial.suggest_int('Akin_min_data_in_leaf', 20, 1000)
#        params_opt['Akin_min_gain_to_split'] = trial.suggest_float('Akin_min_gain_to_split', 0.1, 10.0)
#        params_opt['Akin_path_smooth'] = trial.suggest_float('Akin_path_smooth', 0.2, 0.9)
#        params_opt['Akin_min_sum_hessian_in_leaf'] = trial.suggest_float('Akin_min_sum_hessian_in_leaf', 1e-4, 1.0, log=True)
#        # Run your prediction/analysis routine; assume it returns a score (higher is better)
#        logger.info(f" Params: {params_opt}")
#        score = CollectionModels.AkinDistriML_loadup_analyze(
#            assetspl=assetspl, 
#            loadup_name=akinML_binaries_subsetml_name,
#            test_date=test_date_lag,
#            params=params_opt,
#            logger=logger,
#        )
#        return score
#
#    # Create and run the study
#    optuna.logging.enable_propagation()
#    optuna.logging.disable_default_handler()
#    study = optuna.create_study(direction='maximize')
#    study.optimize(objective, n_trials=200)
#
#    print("Best parameters:", study.best_params)
#    print("Best score:", study.best_value)