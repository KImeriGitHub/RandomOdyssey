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
from scipy import stats

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
    'Akin_feature_max': 1492,
    
    'Akin_itersteps': 2,
    'Akin_pre_weight_truncation': 1,
    'Akin_pre_num_leaves': 185,
    'Akin_pre_num_boost_round': 400,
    'Akin_num_leaves': 81,
    'Akin_num_boost_round': 60,
    
    'Akin_feature_fraction': 0.191528,
    'Akin_top_highest': 10,
    
    'Akin_max_depth': 9, #default = -1
    'Akin_learning_rate': 0.011040, #default = 0.1
    'Akin_min_data_in_leaf': 329, # default = 20
    'Akin_min_gain_to_split': 0.108174, # default = 0
    'Akin_path_smooth': 0.749147, # default = 0
    'Akin_min_sum_hessian_in_leaf': 0.164186, #default = 1e-3
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
#    eval_date = pd.Timestamp(year=2025, month=3, day=7, tz='UTC')
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

#if __name__ == "__main__":
#    lagList = np.array([0, 10, 20, 30, 45, 55, 69, 80, 110, 150, 240, 280, 320, 366, 420, 600])
#    lagList = np.unique(np.random.randint(366*6+150, 366*7+150, 13))
#    
#    lagList = np.array([10, 22, 36, 39, 49, 60, 89, 112, 117, 128,
#     132, 170, 201, 209, 203, 230, 263, 268, 288, 325, 384, 414, 489, 512, 594, 
#     606, 608, 616, 637, 670, 720, 725, 772, 778, 783, 805, 810, 866, 
#     873, 901, 950, 1013, 1037, 1043, 1060, 1064, 1098, 1153, 1172, 
#     1242, 1254, 1298, 1331, 1337, 1341, 1360, 1404, 1421, 1452, 1487, 
#     1509, 1546, 1556, 1598, 1620, 1660, 1706, 1728, 1734, 1760, 1775, 
#     1802, 1817, 1889, 1930, 1993, 2025, 2050, 2043, 2085, 2122, 2201, 2278, 2326, 2387, 
#     2411, 2461, 2471, 2500, 2505, 2517, 2536])
#    eval_date = pd.Timestamp(year=2024, month=12, day=13, tz='UTC')
#    res_return = []
#    res_pred = []
#    res_all = []
#    res_allPred = []
#    res_midend = []
#    startTime = datetime.now()
#    for dayLag in lagList:
#        test_date_lag = eval_date - pd.Timedelta(days=dayLag)
#        
#        formatted_date = test_date_lag.strftime('%d%b%Y')
#        
#        akinML_binaries_subsetml_name = (
#            f"AkinDistriML_{stock_group}_{formatted_date}_10days"
#        )
#        
#        logger.info(f"----------Date: {test_date_lag} , Lag: {dayLag}----------")
#        logger.info(f"----------{akinML_binaries_subsetml_name}----------")
#        filePath = os.path.join("src/predictionModule/bin", akinML_binaries_subsetml_name + '.pkl')
#        if not os.path.exists(filePath):
#            CollectionModels.AkinDistriML_saveData(
#                assetspl = assetspl, 
#                save_name = akinML_binaries_subsetml_name,
#                params = params,
#                test_date = test_date_lag,
#                logger = logger,
#            ) 
#        
#        res_loc = CollectionModels.AkinDistriML_loadup_analyze(
#            assetspl = assetspl, 
#            loadup_name = akinML_binaries_subsetml_name,
#            test_date = test_date_lag,
#            params = params,
#            logger = logger,
#        )
#        res_return.append(res_loc[0])
#        res_pred.append(res_loc[1])
#        res_all.append(res_loc[2])
#        res_allPred.append(res_loc[3])
#        res_midend.extend(res_loc[4])
#    
#    endTime = datetime.now()
#    logger.info(f"Time taken: {endTime - startTime}")
#    logger.info("")
#    logger.info(f"Resulting returns: {res_return}")
#    logger.info(f"Resulting returns mean: {np.mean([x for x in res_return if x is not None])}")
#    logger.info(f"Resulting returns variance: {np.var([x for x in res_return if x is not None])}")
#    logger.info("")
#    logger.info(f"Resulting predictions: {res_pred}")
#    logger.info(f"Resulting predictions mean: {np.mean([x for x in res_pred if x is not None])}")
#    logger.info(f"Resulting predictions variance: {np.var([x for x in res_pred if x is not None])}")
#    logger.info("")
#    logger.info(f"Resulting all test stock: {res_all}")
#    logger.info(f"Resulting all test stock mean: {np.mean([x for x in res_all if x is not None])}")
#    logger.info(f"Resulting all test stock variance: {np.var([x for x in res_all if x is not None])}")
#    logger.info("")
#    logger.info(f"Resulting all test predictions: {res_allPred}")
#    logger.info(f"Resulting all test predictions mean: {np.mean([x for x in res_allPred if x is not None])}")
#    logger.info(f"Resulting all test predictions variance: {np.var([x for x in res_allPred if x is not None])}")
#    logger.info("")
#    
#    # Ratio Analysis
#    logger.info("Ratio Analysis")
#    
#    mid_ratios, end_ratios = zip(*res_midend)
#    logger.info(f"Resulting mid ratios: {list(mid_ratios)}")
#    logger.info(f"Resulting end ratios: {list(end_ratios)}")
#
#    # Function to calculate probability based on a given condition
#    def probability(group, condition):
#        return sum(1 for a, b in group if condition(a, b)) / len(group) if group else 0
#
#    # Group the data based on mid ratio thresholds
#    groups = {
#        "mid ratio < 0.995": [pair for pair in res_midend if pair[0] < 0.995],
#        "0.995 <= mid ratio <= 1.005": [pair for pair in res_midend if 0.995 <= pair[0] <= 1.005],
#        "mid ratio > 1.005": [pair for pair in res_midend if pair[0] > 1.005]
#    }
#
#    # Log probabilities where end ratio > mid ratio
#    logger.info("Probabilities for end ratio > mid ratio:")
#    for label, group in groups.items():
#        prob = probability(group, lambda a, b: b > a)
#        logger.info(f"    {label}: {prob}")
#
#    # Log probabilities where end ratio <= mid ratio
#    logger.info("Probabilities for end ratio <= mid ratio:")
#    for label, group in groups.items():
#        prob = probability(group, lambda a, b: b <= a)
#        logger.info(f"    {label}: {prob}")    
#    
#    # Statistics Prediction to Return
#    logger.info("Statictical Analysis All Prediction to top Return")
#    # print statistic like regression for res_pred to res_return
#    # Calculate the correlation coefficient and the p-value
#    correlation, p_value = stats.pearsonr(res_allPred, res_return)
#
#    # Log the results
#    logger.info(f"Correlation coefficient: {correlation}")
#
#    # Perform a linear regression
#    slope, intercept, r_value, p_value, std_err = stats.linregress(res_allPred, res_return)
#
#    # Log the regression results
#    logger.info(f"Linear regression slope: {slope}")
#    logger.info(f"Linear regression intercept: {intercept}")
#    logger.info(f"R-squared: {r_value**2}")
#    logger.info(f"P-value: {p_value}")
#    logger.info(f"Standard error: {std_err}")
#    
#    # Variance of the residuals
#    residuals = np.array(res_return) - (slope * np.array(res_allPred) + intercept)
#    residuals_variance = np.var(residuals)
#    logger.info(f"Variance of the residuals: {residuals_variance}") 
    
if __name__ == "__main__":
    eval_date = pd.Timestamp(year=2024, month=12, day=13, tz='UTC')
    res = []      
    test_date_lag1 = eval_date - pd.Timedelta(days=10)
    test_date_lag2 = eval_date - pd.Timedelta(days=873)
    test_date_lag3 = eval_date - pd.Timedelta(days=1802)
    test_date_lag4 = eval_date - pd.Timedelta(days=2471)
    formatted_date1 = test_date_lag1.strftime('%d%b%Y')
    formatted_date2 = test_date_lag2.strftime('%d%b%Y')
    formatted_date3 = test_date_lag3.strftime('%d%b%Y')
    formatted_date4 = test_date_lag4.strftime('%d%b%Y')
    
    akinML_binaries_subsetml_name1 = (
        f"AkinDistriML_{stock_group}_{formatted_date1}_10days")
    akinML_binaries_subsetml_name2 = (
        f"AkinDistriML_{stock_group}_{formatted_date2}_10days")
    akinML_binaries_subsetml_name3 = (
        f"AkinDistriML_{stock_group}_{formatted_date3}_10days")
    akinML_binaries_subsetml_name4 = (
        f"AkinDistriML_{stock_group}_{formatted_date4}_10days")
    
    def objective(trial):
        # Create a copy of the base params and override the ones to be optimized
        params_opt = params.copy()
        params_opt['Akin_feature_max'] = trial.suggest_int('Akin_feature_max', 300, 2000)
        params_opt['Akin_pre_num_leaves'] = trial.suggest_int('Akin_pre_num_leaves', 50, 200)
        params_opt['Akin_num_leaves'] = trial.suggest_int('Akin_num_leaves', 50, 200)
        params_opt['Akin_itersteps'] = trial.suggest_int('Akin_itersteps', 0, 2)
        params_opt['Akin_pre_weight_truncation'] = trial.suggest_int('Akin_pre_weight_truncation', 1, 2)
        params_opt['Akin_feature_fraction'] = trial.suggest_float('Akin_feature_fraction', 0.01, 0.8)
        params_opt['Akin_max_depth'] = trial.suggest_int('Akin_max_depth', 5, 13)
        params_opt['Akin_learning_rate'] = trial.suggest_float('Akin_learning_rate', 0.005, 0.03, log=True)
        params_opt['Akin_min_data_in_leaf'] = trial.suggest_int('Akin_min_data_in_leaf', 80, 500)
        params_opt['Akin_min_gain_to_split'] = trial.suggest_float('Akin_min_gain_to_split', 0.1, 10.0)
        params_opt['Akin_path_smooth'] = trial.suggest_float('Akin_path_smooth', 0.3, 0.9)
        params_opt['Akin_min_sum_hessian_in_leaf'] = trial.suggest_float('Akin_min_sum_hessian_in_leaf', 1e-4, 1.0, log=True)
        # Run your prediction/analysis routine; assume it returns a score (higher is better)
        logger.info(f" Params: {params_opt}")
        
        score1 = CollectionModels.AkinDistriML_loadup_analyze(
            assetspl=assetspl, 
            loadup_name=akinML_binaries_subsetml_name1,
            test_date=test_date_lag1,
            params=params_opt,
            logger=logger,
        )
        score1 = (score1[0] - (-0.035))/0.035
        
        score2 = CollectionModels.AkinDistriML_loadup_analyze(
            assetspl=assetspl, 
            loadup_name=akinML_binaries_subsetml_name2,
            test_date=test_date_lag2,
            params=params_opt,
            logger=logger,
        )
        score2 = (score2[0] - 0.04)/0.04
        
        score3 = CollectionModels.AkinDistriML_loadup_analyze(
            assetspl=assetspl, 
            loadup_name=akinML_binaries_subsetml_name3,
            test_date=test_date_lag3,
            params=params_opt,
            logger=logger,
        )
        score3 = (score3[0] - 0.005)/0.005
        
        score4 = CollectionModels.AkinDistriML_loadup_analyze(
            assetspl=assetspl, 
            loadup_name=akinML_binaries_subsetml_name4,
            test_date=test_date_lag4,
            params=params_opt,
            logger=logger,
        )
        score4 = (score4[0] - (-0.05))/0.05
        
        return (score1 + score2 + score3 + score4) / 4.0

    # Create and run the study
    optuna.logging.enable_propagation()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=35)

    logger.info("Best parameters:", study.best_params)
    logger.info("Best score:", study.best_value)
    
    df = study.trials_dataframe()
    logger.info("\nTrials DataFrame:")
    logger.info(df.to_string())

    param_importances = optuna.importance.get_param_importances(study)
    logger.info("Parameter Importances:")
    for key, value in param_importances.items():
        logger.info(f"{key}: {value}")