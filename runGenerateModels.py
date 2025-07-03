from src.predictionModule.TreeTimeML_save import TreeTimeML

import pandas as pd
import numpy as np
import polars as pl
import datetime
import secrets
import copy
import logging
import optuna
from optuna.exceptions import TrialPruned
import argparse

stock_group = "group_snp500_finanTo2011"
stock_group_short = "snp500_finanTo2011"
    
params = {
    "daysAfterPrediction": 14,
    'timesteps': 16,
    
    'target_option': 'last',

    "TreeTime_isFiltered": False,
    "TreeTime_FourierRSME_q": None,
    "TreeTime_trend_stc_q": None,
    "TreeTime_trend_mass_index_q": None,
    "TreeTime_AvgReturnPct_qup": None, #0.9616684094957099,
    "TreeTime_volatility_atr_qup": None,
    "TreeTime_ReturnLog_RSMECoeff_2_qup": None,
    "TreeTime_Drawdown_q": None,

    'TreeTime_run_lstm': False,
    "TreeTime_lstm_units": 128,
    "TreeTime_lstm_num_layers": 5,
    "TreeTime_lstm_dropout": 0.001,
    "TreeTime_lstm_recurrent_dropout": 0.002,
    "TreeTime_lstm_learning_rate": 0.02,
    "TreeTime_lstm_optimizer": "rmsprop",
    "TreeTime_lstm_bidirectional": True,
    "TreeTime_lstm_batch_size": 2**11,
    "TreeTime_lstm_epochs": 15,
    "TreeTime_lstm_l1": 0.05,
    "TreeTime_lstm_l2": 0.5,
    "TreeTime_inter_dropout": 0.003,
    "TreeTime_input_gaussian_noise": 0.0005,
    "TreeTime_lstm_conv1d": True,
    "TreeTime_lstm_conv1d_kernel_size": 3,
    "TreeTime_lstm_loss": "quantile_8",
    
    "TreeTime_lgb_num_boost_round": 1500,
    "TreeTime_lgb_lambda_l1": 0.04614242128149404,
    "TreeTime_lgb_lambda_l2": 0.009786276249261908,
    "TreeTime_lgb_feature_fraction": 0.20813359498274574,
    "TreeTime_lgb_num_leaves": 182,
    "TreeTime_lgb_max_depth": 11,
    "TreeTime_lgb_learning_rate": 0.02887444408582576,
    "TreeTime_lgb_min_data_in_leaf": 272,
    "TreeTime_lgb_min_gain_to_split": 0.10066457576238419,
    "TreeTime_lgb_path_smooth": 0.5935679203578974,
    "TreeTime_lgb_min_sum_hessian_in_leaf": 0.3732876155751053,
    "TreeTime_lgb_max_bin": 150,
    
    'TreeTime_lgb_test_size_pct': 0.01,
    
    'TreeTime_MatchFeatures_run': True,
    'TreeTime_MatchFeatures_minWeight': 0.3,
    'TreeTime_MatchFeatures_truncation': 2,
    
    'TreeTime_MatchFeatures_Pricediff': True,
    'TreeTime_MatchFeatures_FinData_quar': False,
    'TreeTime_MatchFeatures_FinData_metrics': False,
    'TreeTime_MatchFeatures_Fourier_RSME': False,
    'TreeTime_MatchFeatures_Fourier_Sign': False,
    'TreeTime_MatchFeatures_TA_trend': True,
    'TreeTime_MatchFeatures_FeatureGroup_VolGrLvl': False,
    'TreeTime_MatchFeatures_LSTM_Prediction': False,
    
    "TreeTime_top_highest": 5,
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
    eval_date = datetime.date(year=2025, month=4, day=16)
    start_train_date = datetime.date(year=2014, month=1, day=1)
    
    testlagList = np.linspace(0, 20, 5).astype(int).tolist()
    evallagList = np.linspace(0, 365*4, 50).astype(int).tolist()
    
    formatted_date = eval_date.strftime('%d%b%Y')
        
    logger.info(f"----------Last Date: {eval_date}----------")
    res = []
    res_1 = []
    for e_date in [eval_date - pd.Timedelta(days=dayLag) for dayLag in evallagList]:
        starttime = datetime.datetime.now()
        treetimeML = TreeTimeML(
            train_start_date=start_train_date,
            test_dates=[e_date], #[e_date - pd.Timedelta(days=dayLag) for dayLag in testlagList],
            group=stock_group,
            params=params,
        )
        treetimeML.load_and_filter_sets()
        try:
            res_arr = treetimeML.analyze()
        except Exception as e:
            logger.error(f"Error during analysis for date {e_date}: {e}")
            continue
        res.append(res_arr[0])
        res_1.append(res_arr[1])
    
        logger.info(f"All results: {res}")
        logger.info(f"Mean over all: {np.mean(res)}") 
        logger.info(f"All predictions: {res_1}")
        logger.info(f"Mean predictions over all: {np.mean(res_1)}")
        
    logger.info(f"Time taken for analysis: {datetime.datetime.now() - starttime}")


###########################
## HYPERPARAMETER TUNING ##
###########################
#if __name__ == "__main__":
#    start_train_date = datetime.date(year=2014, month=1, day=1)
#    eval_dates = [
#        datetime.date(year=2025, month=3, day=3),
#        datetime.date(year=2025, month=1, day=3),
#        datetime.date(year=2024, month=8, day=7),
#        datetime.date(year=2024, month=5, day=7),
#        datetime.date(year=2024, month=2, day=7),
#        datetime.date(year=2023, month=12, day=11),
#        datetime.date(year=2023, month=7, day=11),
#        datetime.date(year=2023, month=2, day=11),
#        datetime.date(year=2022, month=9, day=15),
#        datetime.date(year=2021, month=11, day=19),
#    ]
#    
#    formatted_date = datetime.datetime.now().strftime('%d%b%Y')
#    hex_code = secrets.token_hex(4)
#    optuna_study_name = "optuna_study_25Jun2025_ae6e53d7" #f"optuna_study_{formatted_date}_{hex_code}"
#    optuna_duration = 60 * 60 * 10
#    logger.info(f"----------OPTUNA HYPERPARAMETER TUNING for {optuna_study_name}----------")
#    
#    n_eval = 5
#    lagList = np.linspace(0, 20, n_eval).astype(int).tolist()
#    
#    logger.info(f"----------Last Date: {max(eval_dates)}, First Date: {min(eval_dates)}, Amount: {len(eval_dates)}----------")
#    
#    treetime_instances = [None] * len(eval_dates)
#    for i in range(len(eval_dates)):
#        treetime_instance = TreeTimeML(
#            train_start_date = start_train_date,
#            test_dates = [eval_dates[i] - pd.Timedelta(days=lag) for lag in lagList][::-1],
#            group = stock_group,
#            params = params,
#        )
#        treetime_instance.load_and_filter_sets()
#        treetime_instances[i] = treetime_instance
#    
#    def objective(trial: optuna.Trial):
#        # Create a copy of the base params and override the ones to be optimized
#        params_opt = params.copy()
#        # core
#        #params_opt['daysAfterPrediction'] = trial.suggest_categorical('daysAfterPrediction', [5,7])
#        #params_opt['timesteps'] = trial.suggest_int('timesteps', 8, 48, step=4)
#        #params_opt['TreeTime_FourierRSME_q'] = trial.suggest_float('TreeTime_FourierRSME_q', 0.01, 0.10)
#        #params_opt['TreeTime_FourierRSME_q'] = None if params_opt['TreeTime_FourierRSME_q'] < 0.09 else params_opt['TreeTime_FourierRSME_q']
#        
#        #params_opt['TreeTime_trend_stc_q'] = trial.suggest_float('TreeTime_trend_stc_q', 0.01, 0.1)
#        #params_opt['TreeTime_trend_stc_q'] = None if params_opt['TreeTime_trend_stc_q'] < 0.09 else params_opt['TreeTime_trend_stc_q']
#        
#        #params_opt['TreeTime_trend_mass_index_q'] = trial.suggest_float('TreeTime_trend_mass_index_q', 0.01, 0.1)
#        #params_opt['TreeTime_trend_mass_index_q'] = None if params_opt['TreeTime_trend_mass_index_q'] < 0.09 else params_opt['TreeTime_trend_mass_index_q']
#        
#        #params_opt['TreeTime_AvgReturnPct_qup'] = trial.suggest_float('TreeTime_AvgReturnPct_qup', 0.9, 0.99)
#        #params_opt['TreeTime_AvgReturnPct_qup'] = None if params_opt['TreeTime_AvgReturnPct_qup'] > 0.995 else params_opt['TreeTime_AvgReturnPct_qup']
#        
#        #params_opt['TreeTime_volatility_atr_qup'] = trial.suggest_float('TreeTime_volatility_atr_qup', 0.9, 0.99)
#        #params_opt['TreeTime_volatility_atr_qup'] = 0.01 if params_opt['TreeTime_volatility_atr_qup'] > 0.995 else params_opt['TreeTime_volatility_atr_qup']
#        
#        #params_opt['TreeTime_ReturnLog_RSMECoeff_2_qup'] = trial.suggest_float('TreeTime_ReturnLog_RSMECoeff_2_qup', 0.9, 0.99)
#        #params_opt['TreeTime_ReturnLog_RSMECoeff_2_qup'] = None if params_opt['TreeTime_ReturnLog_RSMECoeff_2_qup'] > 0.995 else params_opt['TreeTime_ReturnLog_RSMECoeff_2_qup']
#        
#        #params_opt['TreeTime_Drawdown_q'] = trial.suggest_float('TreeTime_Drawdown_q', 0.01, 0.1)
#        #params_opt['TreeTime_Drawdown_q'] = None if params_opt['TreeTime_Drawdown_q'] < 0.09 else params_opt['TreeTime_Drawdown_q']
#
#        # LSTM
#        params_opt['TreeTime_run_lstm'] = False #trial.suggest_categorical('TreeTime_run_lstm', [True, False])
#        #params_opt['TreeTime_lstm_units'] = trial.suggest_int('TreeTime_lstm_units', 32, 128, step=32) 
#        #params_opt['TreeTime_lstm_num_layers'] = trial.suggest_int('TreeTime_lstm_num_layers', 1, 6)
#        #params_opt['TreeTime_lstm_dropout'] = trial.suggest_float('TreeTime_lstm_dropout', 0.0001, 0.3, log=True)
#        #params_opt['TreeTime_lstm_recurrent_dropout'] = trial.suggest_float('TreeTime_lstm_recurrent_dropout', 0.0001, 0.3, log=True)
#        #params_opt['TreeTime_lstm_learning_rate'] = trial.suggest_float('TreeTime_lstm_learning_rate', 0.0001, 1.2, log=True)
#        #params_opt['TreeTime_lstm_optimizer'] = trial.suggest_categorical('TreeTime_lstm_optimizer',['adam', 'rmsprop'])
#        #params_opt['TreeTime_lstm_bidirectional'] = trial.suggest_categorical('TreeTime_lstm_bidirectional',[True, False])
#        #params_opt['TreeTime_lstm_batch_size'] = int(np.pow(2,trial.suggest_int('TreeTime_lstm_batch_size_pow', 8, 12)))
#        #params_opt['TreeTime_lstm_epochs'] = trial.suggest_int('TreeTime_lstm_epochs', 6, 96, step=10)
#        #params_opt['TreeTime_lstm_l1'] = trial.suggest_float('TreeTime_lstm_l1', 0.01, 6.0, log=True)
#        #params_opt['TreeTime_lstm_l2'] = trial.suggest_float('TreeTime_lstm_l2', 0.01, 6.0, log=True)
#        #params_opt['TreeTime_inter_dropout'] = trial.suggest_float('TreeTime_inter_dropout', 0.001, 0.3, log=True)
#        #params_opt['TreeTime_input_gaussian_noise'] = trial.suggest_float('TreeTime_input_gaussian_noise', 0.0001, 0.2, log=True)
#        #params_opt["TreeTime_lstm_conv1d_kernel_size"] = trial.suggest_int('TreeTime_lstm_conv1d_kernel_size', 0, 9)
#        #params_opt["TreeTime_lstm_conv1d"] = True if params_opt["TreeTime_lstm_conv1d_kernel_size"] > 0 else False
#        #params_opt['TreeTime_lstm_loss'] = trial.suggest_categorical('TreeTime_lstm_loss', ['mse', 'quantile_2', 'quantile_5', 'quantile_8', 'r2'])
#
#        # LightGBM
#        params_opt['TreeTime_lgb_num_boost_round'] = 1500; #trial.suggest_int('TreeTime_lgb_num_boost_round', 500, 3500, step=500)
#        params_opt['TreeTime_lgb_lambda_l1'] = trial.suggest_float('TreeTime_lgb_lambda_l1', 0.001, 0.5, log=True)
#        params_opt['TreeTime_lgb_lambda_l2'] = trial.suggest_float('TreeTime_lgb_lambda_l2', 0.001, 0.5, log=True)
#        params_opt['TreeTime_lgb_feature_fraction'] = trial.suggest_float('TreeTime_lgb_feature_fraction', 0.005, 0.05, log=False)
#        params_opt['TreeTime_lgb_max_depth'] = trial.suggest_int('TreeTime_lgb_max_depth', 2, 9)
#        params_opt['TreeTime_lgb_num_leaves'] = min(int((2**params_opt['TreeTime_lgb_max_depth']) * trial.suggest_float('TreeTime_lgb_num_leaves_rat', 0.2, 1.3)), 800)
#        params_opt['TreeTime_lgb_learning_rate'] = trial.suggest_float('TreeTime_lgb_learning_rate', 0.0001, 0.03, log=True)
#        #params_opt['TreeTime_lgb_min_data_in_leaf'] = trial.suggest_int('TreeTime_lgb_min_data_in_leaf', 30, 430, step=20)
#        #params_opt['TreeTime_lgb_min_gain_to_split'] = trial.suggest_float('TreeTime_lgb_min_gain_to_split', 0.001, 2.9, log=True)
#        #params_opt['TreeTime_lgb_path_smooth'] = trial.suggest_float('TreeTime_lgb_path_smooth', 0.1, 0.9)
#        #params_opt['TreeTime_lgb_min_sum_hessian_in_leaf'] = trial.suggest_float('TreeTime_lgb_min_sum_hessian_in_leaf', 0.01, 10.0, log=True)
#        #params_opt['TreeTime_lgb_max_bin'] = trial.suggest_int('TreeTime_lgb_max_bin', 100, 1500, step=100)
#        params_opt['TreeTime_lgb_test_size_pct'] = 0.01
#        
#        # MatchFeatures
#        params_opt['TreeTime_MatchFeatures_run'] = False #trial.suggest_categorical('TreeTime_MatchFeatures_run', [True, False])
#        #params_opt['TreeTime_MatchFeatures_minWeight'] = trial.suggest_float('TreeTime_MatchFeatures_minWeight', 0.1, 0.8)
#        
#        # Run your prediction/analysis routine; assume it returns a score (higher is better)
#        logger.info(f" Params:")
#        res_arr_list = [None] * len(eval_dates)
#        for key, value in params_opt.items():
#            logger.info(f"  {key}: {value}")
#        try:    
#            for i, _ in enumerate(eval_dates):
#                treetime_instance: TreeTimeML = copy.deepcopy(treetime_instances[i])
#                res_arr_list[i] = treetime_instance.analyze(params=params_opt)
#        except Exception as e:
#            # optional: record the error message
#            trial.set_user_attr("error", str(e))
#            # prune this trial
#            raise TrialPruned()
#        
#        res_arr_0 = np.mean([res[0] for res in res_arr_list if res is not None])
#        res_arr_1 = np.mean([res[1] for res in res_arr_list if res is not None])
#        res_arr_2 = np.mean([res[2] for res in res_arr_list if res is not None])
#        res_arr_3 = np.mean([res[3] for res in res_arr_list if res is not None])
#        res_arr_4 = np.mean([res[4] for res in res_arr_list if res is not None])
#        
#        logger.info(f"Model actual mean return over dates: {[res[0] for res in res_arr_list if res is not None]} with mean {res_arr_0}")   
#        logger.info(f"Model train NDCG@1 score over dates: {[res[1] for res in res_arr_list if res is not None]} with mean {res_arr_1}")
#        logger.info(f"Model train NDCG@5 score over dates: {[res[2] for res in res_arr_list if res is not None]} with mean {res_arr_2}")
#        logger.info(f"Model test NDCG@1  score over dates: {[res[3] for res in res_arr_list if res is not None]} with mean {res_arr_3}")
#        logger.info(f"Model test NDCG@5  score over dates: {[res[4] for res in res_arr_list if res is not None]} with mean {res_arr_4}")
#        score = res_arr_4
#        
#        return score
#
#    # Create and run the study
#    optuna.logging.enable_propagation()
#    study = optuna.create_study(
#        study_name=optuna_study_name,
#        storage="sqlite:///optuna.db",
#        direction="maximize",
#        load_if_exists=True              # don’t error if it already exists
#    )
#    study.optimize(objective, timeout=optuna_duration)
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
#    p = argparse.ArgumentParser()
#    p.add_argument('--year',  type=int, default=2025, help='Year (default: 2025)')
#    p.add_argument('--month', type=int, required=True, help='Month as a number (1–12)')
#    p.add_argument('--day',   type=int, required=True, help='Day as a number (1–31)')
#    args = p.parse_args()
#
#    eval_date = datetime.date(year=args.year, month=args.month, day=args.day)
#    
#    start_train_date = datetime.date(year=2014, month=1, day=1)
#    formatted_date = eval_date.strftime('%d%b%Y')
#    
#    lagList = np.linspace(0, 180, 50).astype(int).tolist()
#    test_dates = (
#        [eval_date - pd.Timedelta(days=dayLag) for dayLag in range(0, params['daysAfterPrediction'] + 1)] 
#        + [eval_date - pd.Timedelta(days=dayLag + params['daysAfterPrediction']+1) for dayLag in lagList]
#    )
#    test_dates = test_dates[::-1]  # Reverse the order
#    
#    logger.info(f"----------Last Date: {eval_date}, First Date: {min(test_dates)}, Amount: {len(test_dates)}----------")
#    starttime = datetime.datetime.now()
#    treetimeML = TreeTimeML(
#        train_start_date=start_train_date,
#        test_dates=test_dates,
#        group=stock_group,
#        params=params,
#    )
#    treetimeML.load_and_filter_sets()
#    res_arr = treetimeML.predict()
#    logger.info(f"Time taken for analysis: {datetime.datetime.now() - starttime}")
#    
#    logger.info(f"Model actual mean return over dates: {res_arr[0]}")   
#    logger.info(f"Model predictions mean return over dates: {res_arr[1]}")
#    logger.info("")