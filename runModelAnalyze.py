from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.predictionModule.TreeTimeML import TreeTimeML
from src.predictionModule.LoadupSamples import LoadupSamples
import treetimeParams

import pandas as pd
import numpy as np
import polars as pl
import datetime
import copy
import random

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
final_eval_date   = datetime.date(2025, 8, 7)    # last date you want to consider cutoffs up to
test_horizon_days = 7                             # days after train cutoff for test slice
n_cutoffs = 100                                     # number of cutoffs to generate
num_reruns = 1                                     # number of times to rerun analysis for each cutoff
days_delta = 7                                   # days delta for cutoff generation

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

        pred_float_list = []
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

                pred_float_list.append(res_loop) if res_loop is not None else None
                res_dict_list.append(res_dict_loop) if res_dict_loop is not None else None
            
            pred_float_list = np.array(pred_float_list)
            if pred_float_list.size > 0:
                res_dict = res_dict_list[np.argmax(pred_float_list)] if pred_float_list.size > 0 else None

                logger.info(f"[{end_train_date}] Model actual mean return over dates: {np.max(pred_float_list)}")
                logger.info(f"[{end_train_date}] Time taken for analysis: {elapsed}")

                results.append(
                    {
                        "end_train_date": end_train_date,
                        "end_test_date": end_test_date,
                        "analysis_time": elapsed.total_seconds(),
                        "res_df": res_dict["df_pred_res"]
                    }
                )
        except Exception as e:
            logger.error(f"Error during analysis for cutoff {end_train_date}: {e}")
            continue

    total_elapsed = datetime.datetime.now() - starttime_all
    logger.info(f"Completed {len(results)} rolling backtests in {total_elapsed}.")

    res_list = [res["res_df"] for res in results]

    results_df:pl.DataFrame = ModelAnalyzer.log_test_result_multiple(res_list, last_col="target_ratio")

    results_df.write_parquet(f"output_analysis_df_{formatted_date}.parquet")