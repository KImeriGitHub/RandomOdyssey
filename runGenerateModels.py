from src.predictionModule.TreeTimeML import TreeTimeML
from src.predictionModule.ModelAnalyzer import ModelAnalyzer

import pandas as pd
import numpy as np
import polars as pl
import datetime
import logging
import optuna
import copy

stock_group = "group_snp500_finanTo2011"
stock_group_short = "snp500_finanTo2011"

params = {
    "daysAfterPrediction": 7,
    'timesteps': 32,
    
    'target_option': 'last',

    "TreeTime_isFiltered": True,
    "TreeTime_RSIExt_q": 0.130029,
    "TreeTime_FourierRSME_q": 0.093668,

    "TreeTime_lstm_units": 64,
    "TreeTime_lstm_num_layers": 2,
    "TreeTime_lstm_dropout": 0.0507791,
    "TreeTime_lstm_recurrent_dropout": 0.0339752,
    "TreeTime_lstm_learning_rate": 0.104990,
    "TreeTime_lstm_optimizer": "rmsprop",
    "TreeTime_lstm_bidirectional": True,
    "TreeTime_lstm_batch_size": 64,
    "TreeTime_lstm_epochs": 2,
    "TreeTime_lstm_l1": 2.4558438,
    "TreeTime_lstm_l2": 5.938704,
    "TreeTime_inter_dropout": 0.056293,
    "TreeTime_input_gaussian_noise": 0.005,
    "TreeTime_lstm_conv1d": True,
    "TreeTime_lstm_conv1d_kernel_size": 5,
    "TreeTime_lstm_loss": "quantile",
    
    'TreeTime_lgb_num_boost_round': 200,
    'TreeTime_lgb_lambda_l1': 2.827429,
    'TreeTime_lgb_lambda_l2': 5.523929,
    'TreeTime_lgb_feature_fraction': 0.965268,
    'TreeTime_lgb_num_leaves': 2500,
    'TreeTime_lgb_max_depth': 15,
    'TreeTime_lgb_learning_rate': 0.11038499,
    'TreeTime_lgb_min_data_in_leaf': 50,
    'TreeTime_lgb_min_gain_to_split': 0.111035,
    'TreeTime_lgb_path_smooth': 0.271341,
    'TreeTime_lgb_min_sum_hessian_in_leaf': 0.252593,
    
    'TreeTime_MatchFeatures_minWeight': 0.3,
    'TreeTime_MatchFeatures_truncation': 1,
    
    'TreeTime_MatchFeatures_Pricediff': False,
    'TreeTime_MatchFeatures_FinData_quar': False,
    'TreeTime_MatchFeatures_FinData_metrics': False,
    'TreeTime_MatchFeatures_Fourier_RSME': False,
    'TreeTime_MatchFeatures_Fourier_Sign': False,
    'TreeTime_MatchFeatures_TA_trend': False,
    'TreeTime_MatchFeatures_FeatureGroup_VolGrLvl': False,
    'TreeTime_MatchFeatures_LSTM_Prediction': True,
    
    "TreeTime_top_highest": 10,
}

formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()

logging.basicConfig(
    filename=f'logs/output_TreeTime_{stock_group_short}_{formatted_date}.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)
logger = logging.getLogger(__name__)
logger.info(f" Params: {params}")

###############
## ANALYZING ##
###############
if __name__ == "__main__":
    eval_date = datetime.date(year=2025, month=2, day=20)
    start_train_date = datetime.date(year=2014, month=1, day=1)
    
    res_return = []
    res_pred = []
    startTime = datetime.datetime.now()
    lagList = np.linspace(0, 2000, 100).astype(int).tolist()
    for dayLag in lagList:
        test_date_lag = eval_date - pd.Timedelta(days=dayLag)
        
        formatted_date = test_date_lag.strftime('%d%b%Y')
        
        logger.info(f"----------Date: {test_date_lag} , Lag: {dayLag}----------")
        
        starttime = datetime.datetime.now()
        treetimeML = TreeTimeML(
            train_start_date=start_train_date,
            test_date=test_date_lag,
            group=stock_group,
            params=params,
        )
        treetimeML.load_and_filter_sets()
        res_loc = treetimeML.analyze()
        logger.info(f"Time taken for analysis: {datetime.datetime.now() - starttime}")
        
        res_return.append(res_loc[0])
        res_pred.append(res_loc[1])

        logger.info(f"Mean results so far: {np.mean(np.array(res_return))}")   
        logger.info(f"Mean prediction so far: {np.mean(np.array(res_pred))}")

        if len(res_return) > 3:
            ModelAnalyzer.print_model_results(np.array(res_pred), np.array(res_return))
    
    endTime = datetime.datetime.now()
    logger.info(f"Time taken: {endTime - startTime}")
    logger.info("")
    
    ModelAnalyzer.print_model_results(res_pred, res_return)


###########################
## HYPERPARAMETER TUNING ##
###########################
#if __name__ == "__main__":
#    start_train_date = datetime.date(year=2014, month=1, day=1)
#    eval_date = datetime.date(year=2024, month=12, day=13)
#    
#    test_date_lag1 = eval_date - pd.Timedelta(days=1172)
#    test_date_lag2 = eval_date - pd.Timedelta(days=288)
#    test_date_lag3 = eval_date - pd.Timedelta(days=1509)
#    test_date_lag4 = eval_date - pd.Timedelta(days=2326)
#    
#    test_dates = [test_date_lag1, test_date_lag2, test_date_lag3, test_date_lag4]
#    bias_vector = [0.99, 0.999, 1.01, 1.029]
#    
#    #Preprocessing
#    starttime = datetime.datetime.now()
#    logger.info(f"Starting Preprocessing.")
#    treetime_instances = []
#    for test_date in test_dates:
#        treetime = TreeTimeML(
#            train_start_date = start_train_date,
#            test_date = test_date,
#            group = stock_group,
#            params = params,
#        )
#        treetime.load_and_filter_sets()
#        treetime_instances.append(treetime)
#        
#    logger.info(f"Preprocessing time: {datetime.datetime.now() - starttime}")
#    
#    def objective(trial: optuna.Trial):
#        # Create a copy of the base params and override the ones to be optimized
#        params_opt = params.copy()
#        # core
#        params_opt['timesteps'] = trial.suggest_int('timesteps', 8, 32, step=4)
#        params_opt['TreeTime_RSIExt_q'] = 0.1#trial.suggest_float('TreeTime_RSIExt_q', 0.05, 0.15)
#        params_opt['TreeTime_FourierRSME_q'] = 0.0#trial.suggest_float('TreeTime_FourierRSME_q', 0.05, 0.15)
#
#        # LSTM
#        params_opt['TreeTime_lstm_units'] = trial.suggest_int('TreeTime_lstm_units', 64, 128, step=32)
#        params_opt['TreeTime_lstm_num_layers'] = 2#trial.suggest_int('TreeTime_lstm_num_layers', 1, 2)
#        params_opt['TreeTime_lstm_dropout'] = trial.suggest_float('TreeTime_lstm_dropout', 0.0001, 0.3, log=True)
#        params_opt['TreeTime_lstm_recurrent_dropout'] = trial.suggest_float('TreeTime_lstm_recurrent_dropout', 0.0001, 0.3, log=True)
#        params_opt['TreeTime_lstm_learning_rate'] = trial.suggest_float('TreeTime_lstm_learning_rate', 0.001, 0.2, log=True)
#        params_opt['TreeTime_lstm_optimizer'] = 'rmsprop'
#        params_opt['TreeTime_lstm_bidirectional'] = False
#        params_opt['TreeTime_lstm_batch_size'] = np.pow(2,trial.suggest_int('TreeTime_lstm_batch_size_pow', 3, 9))
#        params_opt['TreeTime_lstm_epochs'] = 2#trial.suggest_int('TreeTime_lstm_epochs', 1, 2)
#        params_opt['TreeTime_lstm_l1'] = trial.suggest_float('TreeTime_lstm_l1', 0.01, 6.0, log=True)
#        params_opt['TreeTime_lstm_l2'] = trial.suggest_float('TreeTime_lstm_l2', 0.01, 6.0, log=True)
#        params_opt['TreeTime_inter_dropout'] = trial.suggest_float('TreeTime_inter_dropout', 0.001, 0.3, log=True)
#        params_opt['TreeTime_input_gaussian_noise'] = trial.suggest_float('TreeTime_input_gaussian_noise', 0.0001, 0.02, log=True)
#        params_opt["TreeTime_lstm_conv1d"] = True,
#        params_opt["TreeTime_lstm_conv1d_kernel_size"] = 5,
#
#        # LightGBM
#        params_opt['TreeTime_lgb_num_boost_round'] = 500
#        params_opt['TreeTime_lgb_lambda_l1'] = trial.suggest_float('TreeTime_lgb_lambda_l1', 0.001, 6.0, log=True)
#        params_opt['TreeTime_lgb_lambda_l2'] = trial.suggest_float('TreeTime_lgb_lambda_l2', 0.001, 6.0, log=True)
#        params_opt['TreeTime_lgb_feature_fraction'] = trial.suggest_float('TreeTime_lgb_feature_fraction', 0.1, 0.99, log=False)
#        params_opt['TreeTime_lgb_num_leaves'] = trial.suggest_int('TreeTime_lgb_num_leaves', 50, 2500, step=100)
#        params_opt['TreeTime_lgb_max_depth'] = trial.suggest_int('TreeTime_lgb_max_depth', 3, 32, step=5)
#        params_opt['TreeTime_lgb_learning_rate'] = trial.suggest_float('TreeTime_lgb_learning_rate', 0.001, 0.2, log=True)
#        params_opt['TreeTime_lgb_min_data_in_leaf'] = trial.suggest_int('TreeTime_lgb_min_data_in_leaf', 30, 430, step=20)
#        params_opt['TreeTime_lgb_min_gain_to_split'] = trial.suggest_float('TreeTime_lgb_min_gain_to_split', 0.001, 0.9, log=True)
#        params_opt['TreeTime_lgb_path_smooth'] = trial.suggest_float('TreeTime_lgb_path_smooth', 0.05, 0.9, log=True)
#        params_opt['TreeTime_lgb_min_sum_hessian_in_leaf'] = trial.suggest_float('TreeTime_lgb_min_sum_hessian_in_leaf', 0.01, 10.0, log=True)
#        
#        # Run your prediction/analysis routine; assume it returns a score (higher is better)
#        logger.info(f" Params: {params_opt}")
#        
#        def run_treetime(idx):
#            treetime = treetime_instances[idx]
#            treetime_copy: TreeTimeML = copy.deepcopy(treetime)
#            treetime_copy.params = params_opt
#            
#            res_loc = treetime_copy.analyze()
#            return res_loc[0]
#        
#        score0 = run_treetime(0) - bias_vector[0]
#        
#        score1 = run_treetime(1) - bias_vector[1]
#        
#        score2 = run_treetime(2) - bias_vector[2]
#        
#        score3 = run_treetime(3) - bias_vector[3]
#        
#        return (score0 + score1 + score2 + score3) / 4.0
#
#    # Create and run the study
#    optuna.logging.enable_propagation()
#    study = optuna.create_study(direction='maximize')
#    study.optimize(objective, timeout=60*60*3)
#
#    logger.info(f"Best parameters: {study.best_params}")
#    logger.info(f"Best score: {study.best_value}")
#    
#    df = study.trials_dataframe()
#    logger.info("\nTrials DataFrame:")
#    logger.info(df.sort_values("value").to_string())
#
#    param_importances = optuna.importance.get_param_importances(study)
#    logger.info("Parameter Importances:")
#    for key, value in param_importances.items():
#        logger.info(f"{key}: {value}")
        
################
## Prediction ##
################
#if __name__ == "__main__":
#    eval_date = datetime.date(year=2025, month=5, day=13)
#    
#    start_train_date = datetime.date(year=2014, month=1, day=1)
#    formatted_date = eval_date.strftime('%d%b%Y')
#    
#    logger.info(f"----------Date: {eval_date} , PREDICTION----------")
#    params['TreeTime_lstm_epochs'] = 3
#    params['TreeTime_lgb_num_boost_round'] = 1000
#    starttime = datetime.datetime.now()
#    treetimeML = TreeTimeML(
#        train_start_date=start_train_date,
#        test_date=eval_date,
#        group=stock_group,
#        params=params,
#    )
#    treetimeML.load_and_filter_sets()
#    res_loc = treetimeML.predict()
#    logger.info(f"Time taken for analysis: {datetime.datetime.now() - starttime}")
#    
#    endTime = datetime.datetime.now()
#    
#    logger.info(f"Time taken: {endTime - starttime}")
#    logger.info("")
#    logger.info(f"Results: {res_loc}")