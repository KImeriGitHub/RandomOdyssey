from src.predictionModule.TreeTimeML import TreeTimeML
from src.predictionModule.LoadupSamples import LoadupSamples

import treetimeParams
import datetime
import polars as pl
import argparse

stock_group = "group_finanTo2011"
stock_group_short = '_'.join(stock_group.split('_')[1:])

import logging
formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f'logs/output_prediction_TreeTime_{stock_group_short}_{formatted_date}.log',
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
eval_date   = datetime.date(2025, 8, 11)          # last date you want to consider cutoffs up to
test_horizon_days = 7                             # days after train cutoff for test slice
n_cutoffs = 100                                    # number of cutoffs to generate
num_reruns = 2                                    # number of times to rerun analysis for each cutoff  
days_delta = 15                                    # days delta for cutoff generation

if __name__ == "__main__":
    #p = argparse.ArgumentParser()
    #p.add_argument('--year',  type=int, default=2025, help='Year (default: 2025)')
    #p.add_argument('--month', type=int, required=True, help='Month as a number (1-12)')
    #p.add_argument('--day',   type=int, required=True, help='Day as a number (1-31)')
    #args = p.parse_args()

    #eval_date = datetime.date(year=args.year, month=args.month, day=args.day)

    test_dates = [eval_date - datetime.timedelta(days=i) for i in range(test_horizon_days)][::-1]

    ls = LoadupSamples(
        train_start_date=global_start_date,
        test_dates=test_dates,
        group=stock_group,
        group_type='Tree',
        params=params,
    )
    ls.load_samples()

    starttime_all = datetime.datetime.now()

    tt = TreeTimeML(
        train_start_date=ls.train_start_date,
        test_dates=ls.test_dates,
        group=stock_group,
        params=params,
        loadup=ls
    )
                
    pred_meanmean, res_dict = tt.predict()

    df_pred: pl.DataFrame = res_dict['df_pred_res']
    df_pred.write_parquet(f'outputs/output_prediction_TreeTime_{stock_group_short}_{formatted_date}.parquet')

    total_elapsed = datetime.datetime.now() - starttime_all
    logger.info(f"Completed in {total_elapsed}.")
