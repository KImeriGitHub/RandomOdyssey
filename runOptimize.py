import datetime

from src.hyperparameterTuning.OptunaTuning import OptunaTuning
from src.predictionModule.LoadupSamples import LoadupSamples
from src.hyperparameterTuning.OptunaClient import OptunaClient
from src.hyperparameterTuning.StratFilterSamples import StratFilterSamples
import treetimeParams


stock_group = "group_finanTo2011"
stock_group_short = stock_group.replace("group_", "")

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
    optuna_study_name = f"Optuna_{stock_group_short}_{formatted_date}"
    optuna_duration = 60 * 60 * 10
    global_start_date = datetime.date(2014, 1, 1)     # earliest data
    final_eval_date   = datetime.date(2025, 7, 15)    # last date you want to consider cutoffs up to
    test_horizon_days = 7                            # days after train cutoff for test slice
    n_splits = 200

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
    
    # Setup Optuna client
    optuna_client = OptunaClient(
        study_name=optuna_study_name,
        n_trials=n_splits,
        timeout=optuna_duration
    )

    # Setup strategy
    strategy = StratFilterSamples(filter_method="lincomb")

    # Create objective function
    objective = optuna_client.make_objective(strategy=strategy)

    # Run Optuna
    optuna_tuner = OptunaTuning(
        client=optuna_client,
        objective=objective
    )
