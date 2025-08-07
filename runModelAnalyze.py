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
stock_group_short = '_'.join(stock_group.split('_')[1:])

import logging
formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f'logs/output_TreeTime_{stock_group_short}_{formatted_date}.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)
logger = logging.getLogger(__name__)

params = treetimeParams.params

logger.info(f" Params: {params}")

###############
## ANALYZING ##
###############
# Static config
global_start_date = datetime.date(2014, 1, 1)     # earliest data
final_eval_date   = datetime.date(2025, 7, 10)    # last date you want to consider cutoffs up to
test_horizon_days = 7                             # days after train cutoff for test slice
n_cutoffs = 10                                    # number of cutoffs to generate
num_reruns = 2                                    # number of times to rerun analysis for each cutoff  
days_delta = 15                                    # days delta for cutoff generation

if __name__ == "__main__":
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
    cutoffs = [final_eval_date - datetime.timedelta(days=test_horizon_days+i*days_delta + random.randint(0, 3)) for i in range(n_cutoffs)][::-1]

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

        res_dict = {}
        res_dict_list = []
        try:
            for _ in range(num_reruns):
                # Train/analyze for this cutoff
                tt = TreeTimeML(
                    train_start_date=lsc.train_start_date,
                    test_dates=lsc.test_dates,
                    group=stock_group,
                    params=params,
                    loadup=lsc
                )
                starttime = datetime.datetime.now()
                res_loop, res_dict_loop = tt.analyze()
                elapsed = datetime.datetime.now() - starttime
                res_dict_list.append(res_dict_loop)
                    
            res_mean_pred_list = np.array([res['mean_pred'] for res in res_dict_list])
            if res_mean_pred_list.size > 0:
                res_mean_pred_list = np.array([res['mean_pred'] for res in res_dict_list])
                res_dict = res_dict_list[np.argmax(res_mean_pred_list)] if res_mean_pred_list.size > 0 else None

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
        except Exception as e:
            logger.error(f"Error during analysis for cutoff {end_train_date}: {e}")
            continue

    total_elapsed = datetime.datetime.now() - starttime_all
    logger.info(f"Completed {len(results)} rolling backtests in {total_elapsed}.")

    results_df = pd.DataFrame(results).sort_values("end_train_date").reset_index(drop=True)
    logger.info(results_df)
    
    logger.info(f"Mean return over all cutoffs: {results_df['mean_return'].mean()}")
    logger.info(f"Max prediction over all cutoffs: {results_df['max_pred'].mean()}")
    logger.info(f"Mean prediction over all cutoffs: {results_df['mean_pred'].mean()}")
    logger.info(f"Total entries over all cutoffs: {results_df['n_entries'].sum()}")
    
    results_df.to_parquet(f"analysis_df_{formatted_date}.parquet", index=False)