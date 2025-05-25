from src.predictionModule.TreeTimeML import TreeTimeML

import pandas as pd
import numpy as np
import polars as pl
import datetime
import secrets
import logging
import optuna
from optuna.exceptions import TrialPruned

stock_group = "group_snp500_finanTo2011"
stock_group_short = "snp500_finanTo2011"

params = {
    "daysAfterPrediction": 7,
    'timesteps': 20,
    
    'target_option': 'last',

    "TreeTime_isFiltered": True,
    "TreeTime_FourierRSME_q": 0.05,
    'TreeTime_trend_stc_q': 0.05,
    'TreeTime_trend_mass_index_q': 0.05,
    'TreeTime_AvgReturnPct_qup': 0.95,
    'TreeTime_volatility_atr_qup': 0.95,
    'TreeTime_ReturnLog_RSMECoeff_2_qup': 0.95,
    'TreeTime_Drawdown_q': 0.05,

    'TreeTime_run_lstm': True,
    'TreeTime_lstm_units': 64, 
    "TreeTime_lstm_num_layers": 1,
    'TreeTime_lstm_dropout': 0.01, 
    'TreeTime_lstm_recurrent_dropout': 0.01, 
    'TreeTime_lstm_learning_rate': 0.0011953535870423434, 
    "TreeTime_lstm_optimizer": "adam",
    "TreeTime_lstm_bidirectional": True,
    "TreeTime_lstm_batch_size": 2**12,
    "TreeTime_lstm_epochs": 5,
    'TreeTime_lstm_l1': 0.01, 
    'TreeTime_lstm_l2': 0.01, 
    'TreeTime_inter_dropout': 0.01, 
    'TreeTime_input_gaussian_noise': 0.0003866652357697727, 
    "TreeTime_lstm_conv1d": False,
    'TreeTime_lstm_conv1d_kernel_size': 3, 
    'TreeTime_lstm_loss': 'r2', 
    
    'TreeTime_lgb_num_boost_round': 200,
    'TreeTime_lgb_lambda_l1': 0.01, 
    'TreeTime_lgb_lambda_l2': 0.001, 
    'TreeTime_lgb_feature_fraction': 0.3,
    'TreeTime_lgb_num_leaves': 300, 
    'TreeTime_lgb_max_depth': 9, 
    'TreeTime_lgb_learning_rate': 0.06593891266667665, 
    'TreeTime_lgb_min_data_in_leaf': 110, 
    'TreeTime_lgb_min_gain_to_split': 0.04, 
    'TreeTime_lgb_path_smooth': 0.6, 
    'TreeTime_lgb_min_sum_hessian_in_leaf': 1.431081405129,
    
    'TreeTime_MatchFeatures_minWeight': 0.3528104145006282,
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
    
    lagList = np.linspace(0, 700, 50).astype(int).tolist()
    test_dates = [eval_date - pd.Timedelta(days=dayLag) for dayLag in lagList]
    
    formatted_date = eval_date.strftime('%d%b%Y')
        
    logger.info(f"----------Last Date: {eval_date}, First Date: {min(test_dates)}, Amount: {len(test_dates)}----------")

    starttime = datetime.datetime.now()
    treetimeML = TreeTimeML(
        train_start_date=start_train_date,
        test_dates=test_dates,
        group=stock_group,
        params=params,
    )
    treetimeML.load_and_filter_sets()
    res_arr = treetimeML.analyze()
    logger.info(f"Time taken for analysis: {datetime.datetime.now() - starttime}")
    
    logger.info(f"Model actual mean return over dates: {res_arr[0]}")   
    logger.info(f"Model predictions mean return over dates: {res_arr[1]}")
    logger.info(f"Model r2 score over dates: {res_arr[2]}")


###########################
## HYPERPARAMETER TUNING ##
###########################
if __name__ == "__main__":
    start_train_date = datetime.date(year=2014, month=1, day=1)
    model_date = datetime.date(year=2024, month=6, day=14)
    eval_date = datetime.date(year=2025, month=3, day=14)
    
    formatted_date = datetime.datetime.now().strftime('%d%b%Y')
    hex_code = secrets.token_hex(4)
    optuna_study_name = f"optuna_study_{formatted_date}_{hex_code}"
    optuna_duration = 60 * 60 * 24 * 14  # 14 days in seconds
    logger.info(f"----------OPTUNA HYPERPARAMETER TUNING for {optuna_study_name}----------")
    
    n_eval = 50
    lagList = np.linspace(0, 365, n_eval).astype(int).tolist()
    test_dates = [eval_date - pd.Timedelta(days=lag) for lag in lagList]
    
    logger.info(f"----------Last Date: {eval_date}, First Date: {min(test_dates)}, Amount: {len(test_dates)}----------")
    
    def objective(trial: optuna.Trial):
        # Create a copy of the base params and override the ones to be optimized
        params_opt = params.copy()
        # core
        params_opt['daysAfterPrediction'] = trial.suggest_categorical('daysAfterPrediction', [3,4,5,6,7,10])
        params_opt['timesteps'] = trial.suggest_int('timesteps', 8, 48, step=4)
        params_opt['TreeTime_FourierRSME_q'] = trial.suggest_float('TreeTime_FourierRSME_q', 0.01, 0.10, log=True)
        params_opt['TreeTime_trend_stc_q'] = trial.suggest_float('TreeTime_trend_stc_q', 0.01, 0.1, log=True)
        params_opt['TreeTime_trend_mass_index_q'] = trial.suggest_float('TreeTime_trend_mass_index_q', 0.01, 0.1, log=True)
        params_opt['TreeTime_AvgReturnPct_qup'] = trial.suggest_float('TreeTime_AvgReturnPct_qup', 0.9, 0.99, log=True)
        params_opt['TreeTime_volatility_atr_qup'] = trial.suggest_float('TreeTime_volatility_atr_qup', 0.9, 0.99, log=True)
        params_opt['TreeTime_ReturnLog_RSMECoeff_2_qup'] = trial.suggest_float('TreeTime_ReturnLog_RSMECoeff_2_qup', 0.9, 0.99, log=True)
        params_opt['TreeTime_Drawdown_q'] = trial.suggest_float('TreeTime_Drawdown_q', 0.01, 0.1, log=True)

        # LSTM
        params_opt['TreeTime_run_lstm'] = trial.suggest_categorical('TreeTime_run_lstm', [True, False])
        params_opt['TreeTime_lstm_units'] = trial.suggest_int('TreeTime_lstm_units', 32, 128, step=32) 
        params_opt['TreeTime_lstm_num_layers'] = trial.suggest_int('TreeTime_lstm_num_layers', 1, 6)
        params_opt['TreeTime_lstm_dropout'] = trial.suggest_float('TreeTime_lstm_dropout', 0.0001, 0.3, log=True)
        params_opt['TreeTime_lstm_recurrent_dropout'] = trial.suggest_float('TreeTime_lstm_recurrent_dropout', 0.0001, 0.3, log=True)
        params_opt['TreeTime_lstm_learning_rate'] = trial.suggest_float('TreeTime_lstm_learning_rate', 0.0001, 1.2, log=True)
        params_opt['TreeTime_lstm_optimizer'] = trial.suggest_categorical('TreeTime_lstm_optimizer',['adam', 'rmsprop'])
        params_opt['TreeTime_lstm_bidirectional'] = trial.suggest_categorical('TreeTime_lstm_bidirectional',[True, False])
        params_opt['TreeTime_lstm_batch_size'] = int(np.pow(2,trial.suggest_int('TreeTime_lstm_batch_size_pow', 8, 12)))
        params_opt['TreeTime_lstm_epochs'] = trial.suggest_int('TreeTime_lstm_epochs', 6, 96, step=10)
        params_opt['TreeTime_lstm_l1'] = trial.suggest_float('TreeTime_lstm_l1', 0.01, 6.0, log=True)
        params_opt['TreeTime_lstm_l2'] = trial.suggest_float('TreeTime_lstm_l2', 0.01, 6.0, log=True)
        params_opt['TreeTime_inter_dropout'] = trial.suggest_float('TreeTime_inter_dropout', 0.001, 0.3, log=True)
        params_opt['TreeTime_input_gaussian_noise'] = trial.suggest_float('TreeTime_input_gaussian_noise', 0.0001, 0.2, log=True)
        params_opt["TreeTime_lstm_conv1d_kernel_size"] = trial.suggest_int('TreeTime_lstm_conv1d_kernel_size', 0, 9)
        params_opt["TreeTime_lstm_conv1d"] = True if params_opt["TreeTime_lstm_conv1d_kernel_size"] > 0 else False
        params_opt['TreeTime_lstm_loss'] = trial.suggest_categorical('TreeTime_lstm_loss', ['mse', 'quantile_2', 'quantile_5', 'quantile_8', 'r2'])

        # LightGBM
        params_opt['TreeTime_lgb_num_boost_round'] = 500
        params_opt['TreeTime_lgb_lambda_l1'] = trial.suggest_float('TreeTime_lgb_lambda_l1', 0.001, 6.0, log=True)
        params_opt['TreeTime_lgb_lambda_l2'] = trial.suggest_float('TreeTime_lgb_lambda_l2', 0.001, 6.0, log=True)
        params_opt['TreeTime_lgb_feature_fraction'] = trial.suggest_float('TreeTime_lgb_feature_fraction', 0.1, 0.99, log=False)
        params_opt['TreeTime_lgb_num_leaves'] = trial.suggest_int('TreeTime_lgb_num_leaves', 200, 4500, step=100)
        params_opt['TreeTime_lgb_max_depth'] = trial.suggest_int('TreeTime_lgb_max_depth', 13, 43, step=5)
        params_opt['TreeTime_lgb_learning_rate'] = trial.suggest_float('TreeTime_lgb_learning_rate', 0.0001, 1.2, log=True)
        params_opt['TreeTime_lgb_min_data_in_leaf'] = trial.suggest_int('TreeTime_lgb_min_data_in_leaf', 30, 430, step=20)
        params_opt['TreeTime_lgb_min_gain_to_split'] = trial.suggest_float('TreeTime_lgb_min_gain_to_split', 0.001, 2.9, log=True)
        params_opt['TreeTime_lgb_path_smooth'] = trial.suggest_float('TreeTime_lgb_path_smooth', 0.05, 0.9, log=True)
        params_opt['TreeTime_lgb_min_sum_hessian_in_leaf'] = trial.suggest_float('TreeTime_lgb_min_sum_hessian_in_leaf', 0.01, 10.0, log=True)
        
        params_opt['TreeTime_MatchFeatures_minWeight'] = trial.suggest_float('TreeTime_MatchFeatures_minWeight', 0.1, 0.8)
        
        # Run your prediction/analysis routine; assume it returns a score (higher is better)
        logger.info(f" Params:")
        for key, value in params_opt.items():
            logger.info(f"  {key}: {value}")
        try:    
            treetime_instance = TreeTimeML(
                train_start_date = start_train_date,
                test_dates = test_dates,
                group = stock_group,
                params = params_opt,
            )
            treetime_instance.load_and_filter_sets()
            res_arr = treetime_instance.analyze(logger_disabled=True)
        except Exception as e:
            # optional: record the error message
            trial.set_user_attr("error", str(e))
            # prune this trial
            raise TrialPruned()
        
        logger.info(f"Model actual mean return over dates: {res_arr[0]}")   
        logger.info(f"Model predictions mean return over dates: {res_arr[1]}")
        logger.info(f"Model r2 score over dates: {res_arr[2]}")
        log_score = np.where(res_arr[0] > 0.0, np.log(res_arr[0]), -1.0)
        return log_score

    # Create and run the study
    optuna.logging.enable_propagation()
    study = optuna.create_study(
        study_name=optuna_study_name,
        storage="sqlite:///optuna.db",
        direction="maximize",
        load_if_exists=True              # don’t error if it already exists
    )
    study.optimize(objective, timeout=optuna_duration)

    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best score: {study.best_value}")
    
    df = study.trials_dataframe()
    logger.info("\nTrials DataFrame:")
    logger.info(df.sort_values("value").to_string())

    param_importances = optuna.importance.get_param_importances(study)
    logger.info("Parameter Importances:")
    for key, value in param_importances.items():
        logger.info(f"{key}: {value}")
        
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
#    
#    starttime = datetime.datetime.now()
#    treetimeML = TreeTimeML(
#        train_start_date=start_train_date,
#        test_dates=[eval_date],
#        group=stock_group,
#        params=params,
#    )
#    treetimeML.load_and_filter_sets()
#    res_arr = treetimeML.analyze()
#    logger.info(f"Time taken for analysis: {datetime.datetime.now() - starttime}")
#    
#    logger.info(f"Model actual mean return over dates: {res_arr[0]}")   
#    logger.info(f"Model predictions mean return over dates: {res_arr[1]}")
#    logger.info(f"Model r2 score over dates: {res_arr[2]}")
#    logger.info("")