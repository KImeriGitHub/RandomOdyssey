from src.predictionModule.TreeTimeML import TreeTimeML
from src.predictionModule.LoadupSamples import LoadupSamples

import pandas as pd
import numpy as np
import polars as pl
import datetime
import secrets
import copy
import random
import logging
import optuna
from optuna.exceptions import TrialPruned
import argparse

stock_group = "group_snp500_finanTo2011"
stock_group_short = "snp500_finanTo2011"
    
params = {
    "idxAfterPrediction": 5,
    'timesteps': 40,
    'target_option': 'last',
    
    "TreeTime_top_n": 5,
    "TreeTime_stoploss": 0.92,
    
    "FilterSamples_days_to_train_end": 500,
    
    "FilterSamples_lincomb_q_up": 0.95,
    "FilterSamples_lincomb_lr": 0.0008,
    "FilterSamples_lincomb_epochs": 500,
    "FilterSamples_lincomb_probs_noise_std": 0.010503627436184224,
    "FilterSamples_lincomb_subsample_ratio": 0.10,
    "FilterSamples_lincomb_sharpness": 0.6,
    "FilterSamples_lincomb_featureratio": 0.33269072850403053,
    "FilterSamples_lincomb_itermax": 1,
    "FilterSamples_lincomb_show_progress": True,

    'TreeTime_run_lstm': False,
    "LSTM_units": 32,
    "LSTM_num_layers": 3,
    "LSTM_dropout": 0.00001,
    "LSTM_recurrent_dropout": 0.00001,
    "LSTM_learning_rate": 0.001,
    "LSTM_optimizer": "adam",
    "LSTM_bidirectional": True,
    "LSTM_batch_size": 2**12,
    "LSTM_epochs": 3,
    "LSTM_l1": 0.0001,
    "LSTM_l2": 0.0001,
    "LSTM_inter_dropout": 0.0001,
    "LSTM_input_gaussian_noise": 0.0001,
    "LSTM_conv1d": True,
    "LSTM_conv1d_kernel_size": 3,
    "LSTM_loss": "mse",
    
    "LGB_num_boost_round": 100,
    "LGB_lambda_l1": 0.04614242128149404,
    "LGB_lambda_l2": 0.009786276249261908,
    "LGB_feature_fraction": 0.20813359498274574,
    "LGB_num_leaves": 180,
    "LGB_max_depth": 6,
    "LGB_learning_rate": 0.02887444408582576,
    "LGB_min_data_in_leaf": 150,
    "LGB_min_gain_to_split": 0.10066457576238419,
    "LGB_path_smooth": 0.5935679203578974,
    "LGB_min_sum_hessian_in_leaf": 0.3732876155751053,
    "LGB_max_bin": 250,
        
    'TreeTime_MatchFeatures_run': True,
    'TreeTime_MatchFeatures_minWeight': 0.4,
    'TreeTime_MatchFeatures_truncation': 2,
    'TreeTime_MatchFeatures_Pricediff': True,
    'TreeTime_MatchFeatures_FinData_quar': False,
    'TreeTime_MatchFeatures_FinData_metrics': False,
    'TreeTime_MatchFeatures_Fourier_RSME': False,
    'TreeTime_MatchFeatures_Fourier_Sign': False,
    'TreeTime_MatchFeatures_TA_trend': False,
    'TreeTime_MatchFeatures_FeatureGroup_VolGrLvl': False,
    'TreeTime_MatchFeatures_LSTM_Prediction': False,
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
    # Static config
    global_start_date = datetime.date(2014, 1, 1)     # earliest data
    final_eval_date   = datetime.date(2025, 2, 10)    # last date you want to consider cutoffs up to
    test_horizon_days = 7                            # days after train cutoff for test slice

    # Pre-load once
    ls = LoadupSamples(
        train_start_date=global_start_date,
        test_dates=[final_eval_date],  # will be overridden in split loop; kept for init
        group=stock_group,
        params=params,
    )
    ls.load_samples(main_path="./src/featureAlchemy/bin/")

    results = []

    # Generate many training cutoff dates (month-end roll). Change freq as desired.
    cutoffs = [datetime.date(2025, 1, 15) - datetime.timedelta(days=i*8*2+random.randint(-3, 3)) for i in range(40)][::-1]

    starttime_all = datetime.datetime.now()

    for end_train_date in cutoffs:
        end_test_date = end_train_date + datetime.timedelta(days=test_horizon_days)
        lsc = copy.deepcopy(ls)  # Create a copy of the LoadupSamples instance
        # Re-split dataset for this window
        lsc.split_dataset(
            start_date=global_start_date,
            last_train_date=end_train_date,
            last_test_date=end_test_date,
        )

        num_reruns = 3
        res_dict = {}
        res_dict_list = []
        for _ in range(num_reruns):
            # Train/analyze for this cutoff
            tt = TreeTimeML(
                train_start_date=lsc.train_start_date,
                test_dates=lsc.test_dates,
                group=stock_group,
                params=params,
                loadup=lsc,
            )
            try:
                starttime = datetime.datetime.now()
                res_dict_loop = tt.analyze()
                elapsed = datetime.datetime.now() - starttime
            except Exception as e:
                logger.error(f"Error during analysis for cutoff {end_train_date}: {e}")
                continue  # Skip to next cutoff if error occurs
            
            res_dict_list.append(res_dict_loop)

        res_mean_pred_list = np.array([res['mean_pred'] for res in res_dict_list])
        res_dict = res_dict_list[np.argmax(res_mean_pred_list)]

        logger.info(f"[{end_train_date}] Model actual mean return over dates: {res_dict['result']}")
        logger.info(f"[{end_train_date}] Time taken for analysis: {elapsed}")

        results.append(
            {
                "end_train_date": end_train_date,
                "end_test_date": end_test_date,
                "mean_return": res_dict['result'],
                "n_entries": res_dict['n_entries'],
                "analysis_time": elapsed.total_seconds(),
                "max_pred": res_dict['max_pred'],
                "mean_pred": res_dict['mean_pred'],
            }
        )

    total_elapsed = datetime.datetime.now() - starttime_all
    logger.info(f"Completed {len(results)} rolling backtests in {total_elapsed}.")

    # Optional: DataFrame of all results
    results_df = pd.DataFrame(results).sort_values("end_train_date").reset_index(drop=True)
    logger.info(results_df)
    
    logger.info(f"Mean return over all cutoffs: {results_df['mean_return'].mean()}")
    logger.info(f"Max prediction over all cutoffs: {results_df['max_pred'].mean()}")
    logger.info(f"Mean prediction over all cutoffs: {results_df['mean_pred'].mean()}")
    logger.info(f"Total entries over all cutoffs: {results_df['n_entries'].sum()}")
    
    results_df.to_parquet(f"analysis_df_{formatted_date}.parquet", index=False)

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
#        #params_opt['LSTM_units'] = trial.suggest_int('LSTM_units', 32, 128, step=32) 
#        #params_opt['LSTM_num_layers'] = trial.suggest_int('LSTM_num_layers', 1, 6)
#        #params_opt['LSTM_dropout'] = trial.suggest_float('LSTM_dropout', 0.0001, 0.3, log=True)
#        #params_opt['LSTM_recurrent_dropout'] = trial.suggest_float('LSTM_recurrent_dropout', 0.0001, 0.3, log=True)
#        #params_opt['LSTM_learning_rate'] = trial.suggest_float('LSTM_learning_rate', 0.0001, 1.2, log=True)
#        #params_opt['LSTM_optimizer'] = trial.suggest_categorical('LSTM_optimizer',['adam', 'rmsprop'])
#        #params_opt['LSTM_bidirectional'] = trial.suggest_categorical('LSTM_bidirectional',[True, False])
#        #params_opt['LSTM_batch_size'] = int(np.pow(2,trial.suggest_int('LSTM_batch_size_pow', 8, 12)))
#        #params_opt['LSTM_epochs'] = trial.suggest_int('LSTM_epochs', 6, 96, step=10)
#        #params_opt['LSTM_l1'] = trial.suggest_float('LSTM_l1', 0.01, 6.0, log=True)
#        #params_opt['LSTM_l2'] = trial.suggest_float('LSTM_l2', 0.01, 6.0, log=True)
#        #params_opt['LSTM_inter_dropout'] = trial.suggest_float('LSTM_inter_dropout', 0.001, 0.3, log=True)
#        #params_opt['LSTM_input_gaussian_noise'] = trial.suggest_float('LSTM_input_gaussian_noise', 0.0001, 0.2, log=True)
#        #params_opt["LSTM_conv1d_kernel_size"] = trial.suggest_int('LSTM_conv1d_kernel_size', 0, 9)
#        #params_opt["LSTM_conv1d"] = True if params_opt["LSTM_conv1d_kernel_size"] > 0 else False
#        #params_opt['LSTM_loss'] = trial.suggest_categorical('LSTM_loss', ['mse', 'quantile_2', 'quantile_5', 'quantile_8', 'r2'])
#
#        # LightGBM
#        params_opt['LGB_num_boost_round'] = 1500; #trial.suggest_int('LGB_num_boost_round', 500, 3500, step=500)
#        params_opt['LGB_lambda_l1'] = trial.suggest_float('LGB_lambda_l1', 0.001, 0.5, log=True)
#        params_opt['LGB_lambda_l2'] = trial.suggest_float('LGB_lambda_l2', 0.001, 0.5, log=True)
#        params_opt['LGB_feature_fraction'] = trial.suggest_float('LGB_feature_fraction', 0.005, 0.05, log=False)
#        params_opt['LGB_max_depth'] = trial.suggest_int('LGB_max_depth', 2, 9)
#        params_opt['LGB_num_leaves'] = min(int((2**params_opt['LGB_max_depth']) * trial.suggest_float('LGB_num_leaves_rat', 0.2, 1.3)), 800)
#        params_opt['LGB_learning_rate'] = trial.suggest_float('LGB_learning_rate', 0.0001, 0.03, log=True)
#        #params_opt['LGB_min_data_in_leaf'] = trial.suggest_int('LGB_min_data_in_leaf', 30, 430, step=20)
#        #params_opt['LGB_min_gain_to_split'] = trial.suggest_float('LGB_min_gain_to_split', 0.001, 2.9, log=True)
#        #params_opt['LGB_path_smooth'] = trial.suggest_float('LGB_path_smooth', 0.1, 0.9)
#        #params_opt['LGB_min_sum_hessian_in_leaf'] = trial.suggest_float('LGB_min_sum_hessian_in_leaf', 0.01, 10.0, log=True)
#        #params_opt['LGB_max_bin'] = trial.suggest_int('LGB_max_bin', 100, 1500, step=100)
#        params_opt['LGB_test_size_pct'] = 0.01
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