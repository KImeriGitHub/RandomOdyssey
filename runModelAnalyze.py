from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.predictionModule.TreeTimeML import TreeTimeML
from src.predictionModule.LoadupSamples import LoadupSamples

import datetime
import random

timegroup = "group_regOHLCV_to2014"
stock_group = "group_dez_lowspread"
stock_group_short = '_'.join(stock_group.split('_')[1:])

import logging
formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f'logs/output_TreeTime_{stock_group_short}_{formatted_date}.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

loadup_params = {
    "daysAfterPrediction": None,
    "idxAfterPrediction": 5,
    'timesteps': 90,
    
    "LoadupSamples_time_inc_factor": 1,
    "LoadupSamples_tree_scaling_standard": False,
    "LoadupSamples_time_scaling_stretch": False,
}

###############
## ANALYZING ##
###############
# Static config
global_start_date = datetime.date(2014, 1, 1)     # earliest data
final_eval_date   = datetime.date(2025, 10, 7)    # last date you want to consider cutoffs up to
test_horizon_days = 7                             # days after train cutoff for test slice
n_splits = 1000                                     # number of cutoffs to generate
days_delta = 4                                    # days delta for cutoff generation

logger.info("Stock group: %s", stock_group)
logger.info("Time group: %s", timegroup)
logger.info("Global start date: %s", global_start_date)
logger.info("Final evaluation date: %s", final_eval_date)
logger.info("Number of splits: %s", n_splits)

if __name__ == "__main__":
    # Pre-load once
    test_dates = [final_eval_date - datetime.timedelta(days=i) for i in range(test_horizon_days)][::-1]
    ls = LoadupSamples(
        train_start_date=global_start_date,
        test_dates=test_dates,  # will be overridden in split loop; kept for init
        treegroup=stock_group,
        timegroup=timegroup,
        params=loadup_params,
    )
    ls.load_samples()
    

    # Generate many training cutoff dates (month-end roll). Change freq as desired.
    cutoffs = [final_eval_date - datetime.timedelta(
        days=test_horizon_days+i*days_delta + random.randint(0, days_delta//3)) for i in range(n_splits)][::-1]

    starttime_all = datetime.datetime.now()

    results = []
    for end_train_date in cutoffs:
        end_test_date = end_train_date + datetime.timedelta(days=test_horizon_days)

        try:
            lsc = ls.copy(deep=True)
            lsc.split_dataset(
                start_date=global_start_date,
                last_train_date=end_train_date,
                last_test_date=end_test_date,
            )
            # Train/analyze for this cutoff
            tt = TreeTimeML(
                train_start_date=lsc.train_start_date,
                test_dates=lsc.test_dates,
                treegroup=stock_group,
                timegroup=timegroup,
                params=loadup_params,
                loadup=lsc,
            )
            starttime = datetime.datetime.now()
            score_loop, res_dict_loop = tt.analyze()
            elapsed = datetime.datetime.now() - starttime
            
            if score_loop is None or res_dict_loop is None:
                logger.warning(f"[{end_train_date}] Analysis returned no results, skipping.")
                continue

            logger.info(f"[{end_train_date}] Time taken for analysis: {elapsed}")

            results.append(
                {
                    "end_train_date": end_train_date,
                    "end_test_date": end_test_date,
                    "analysis_time": elapsed.total_seconds(),
                    "dfs":  res_dict_loop["res_df"]
                }
            )
        except Exception as e:
            logger.error(f"Error during analysis for cutoff {end_train_date}: {e}")
            continue
        
    if results == []:
        logger.error("No successful rolling backtests were completed.")
        raise RuntimeError("No successful rolling backtests were completed.")

    total_elapsed = datetime.datetime.now() - starttime_all
    logger.info(f"Completed {len(cutoffs)} rolling backtests in {total_elapsed}.")

    res_list = [res["dfs"] for res in results]

    score_col = "test_scores"
    tar_col = "collapsed_ratio"
    results_df = ModelAnalyzer.log_test_result_multiple(res_list, score_col = score_col, last_col=tar_col)