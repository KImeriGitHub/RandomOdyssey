from src.predictionModule.TreeTimeML import TreeTimeML
from src.predictionModule.LoadupSamples import LoadupSamples
import treetimeParams

import pandas as pd
import numpy as np
import polars as pl
import datetime
import secrets
import copy
import random
import optuna
from optuna.exceptions import TrialPruned
import argparse

stock_group = "group_finanTo2011"
stock_group_short = "finanTo2011"

import logging
formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f'logs/output_optuna_TreeTime_{stock_group_short}_{formatted_date}.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)
logger = logging.getLogger(__name__)

params = treetimeParams.params

logger.info(f" Params: {params}")

###########################
## HYPERPARAMETER TUNING ##
###########################
if __name__ == "__main__":
    # Static config
    optuna_study_name = f"TreeTime_{stock_group_short}_{formatted_date}"
    optuna_duration = 60 * 60 * 8  # 2 hours
    global_start_date = datetime.date(2014, 1, 1)     # earliest data
    final_eval_date   = datetime.date(2025, 7, 15)    # last date you want to consider cutoffs up to
    test_horizon_days = 7                            # days after train cutoff for test slice
    n_cutoffs = 8

    # Pre-load once
    test_dates = [final_eval_date - datetime.timedelta(days=i) for i in range(test_horizon_days)][::-1]
    ls = LoadupSamples(
        train_start_date=global_start_date,
        test_dates=test_dates,  # will be overridden in split loop; kept for init
        group=stock_group,
        group_type='Tree',
        params=params,
    )
    ls.load_samples()
    
    results = []

    # Generate many training cutoff dates (month-end roll). Change freq as desired.
    cutoffs = [final_eval_date - datetime.timedelta(days=test_horizon_days+i*90+random.randint(-3, 3)) for i in range(n_cutoffs)][::-1]

    def objective(trial: optuna.Trial):
        # copy base params
        p = params.copy()
        # LightGBM
        p['LGB_max_depth'] = trial.suggest_int('LGB_max_depth', 2, 9)
        p['LGB_max_bin'] = trial.suggest_int('LGB_max_bin', 20, 1023)
        p['TreeTime_MatchFeatures_run'] = trial.suggest_categorical('TreeTime_MatchFeatures_run', [True, False])
        p['FilterSamples_lincomb_itermax'] = trial.suggest_int('FilterSamples_lincomb_itermax', 1, 3)
        p['TreeTime_MatchFeatures_truncation'] = trial.suggest_int('TreeTime_MatchFeatures_truncation', 1, 5)
        p['FilterSamples_lincomb_epochs'] = trial.suggest_int('FilterSamples_lincomb_epochs', 50, 1000, step=50)

        scores = []
        for end_train_date in cutoffs:
            end_test_date = end_train_date + datetime.timedelta(days=test_horizon_days)
            lsc = copy.deepcopy(ls)  # Create a copy of the LoadupSamples instance
            # Re-split dataset for this window
            lsc.split_dataset(
                start_date=global_start_date,
                last_train_date=end_train_date,
                last_test_date=end_test_date,
            )
            try:
                tt = TreeTimeML(
                    train_start_date=lsc.train_start_date,
                    test_dates=lsc.test_dates,
                    group=stock_group,
                    params=params,
                    loadup=lsc
                )
                res_loop, _ = tt.analyze()
                scores.append(np.array(res_loop))
            except:
                scores.append(np.array(1.0))

        return float(np.mean(scores))

    optuna.logging.enable_propagation()
    study = optuna.create_study(
        study_name=optuna_study_name,
        storage="sqlite:///optuna.db",
        direction="maximize",
        load_if_exists=True,              # don’t error if it already exists
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(objective, timeout=optuna_duration)

    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best score: {study.best_value}")

    df: pd.DataFrame = study.trials_dataframe()
    logger.info("\nTrials DataFrame:")
    logger.info(df.sort_values("value").to_string())
    df.to_parquet(f"optuna_trials_{stock_group_short}_{formatted_date}.parquet")

    param_importances = optuna.importance.get_param_importances(study)
    logger.info("Parameter Importances:")
    for key, value in param_importances.items():
        logger.info(f"{key}: {value}")