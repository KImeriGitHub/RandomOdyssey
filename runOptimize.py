import datetime
import logging
from pathlib import Path

from src.hyperparameterTuning.OptunaClient import OptunaClient
from src.hyperparameterTuning.OptunaTuning import OptunaTuning
from src.hyperparameterTuning.StratFilterSamples import StratFilterSamples
from src.hyperparameterTuning.StratLGBLeavesTime import StratLGBLeavesTime
from src.hyperparameterTuning.StratLGBLeavesTree import StratLGBLeavesTree
from src.hyperparameterTuning.StratLGBMOnFiltered import StratLGBMOnFiltered
from src.hyperparameterTuning.StratPCAandLeaves import StratPCAandLeaves
from src.hyperparameterTuning.StratClusteringLSTM import StratClusteringLSTM
from src.hyperparameterTuning.StratRidgeRegression import StratRidgeRegression
from src.predictionModule.LoadupSamples import LoadupSamples

import treetimeParams

timegroup = "group_regOHLCV_over5years"
stock_group = "group_debug"
stock_group_short = '_'.join(stock_group.split('_')[1:])

formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f"logs/output_optuna_{stock_group_short}_{formatted_date}.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)

loadup_params = {
    "idxAfterPrediction": 5,
    'timesteps': 85,
    "LoadupSamples_tree_scaling_standard": False,
    "LoadupSamples_time_scaling_stretch": False,
    "LoadupSamples_time_inc_factor": 1,
}
logger.info("Params: %s", loadup_params)

strategy = StratLGBLeavesTime()
logger.info("Using strategy: %s", StratLGBLeavesTime.__name__)

logger.setLevel(logging.DEBUG)
optuna_study_name = f"Optuna_{stock_group_short}_{formatted_date}"
optuna_duration = 60 * 60
global_start_date = datetime.date(2016, 1, 1)
final_eval_date = datetime.date(2025, 7, 15)
n_test_days = 7
n_splits = 200
n_startup_trials = 3
n_training_days_reserved = 1500
direction = "maximize"

logger.info("Loadup params:")
for k, v in loadup_params.items():
    logger.info("  %s: %s", k, v)
    
logger.info("Precomputing params:")
for k, v in strategy.precompute_params.items():
    logger.info("  %s: %s", k, v)

logger.info("Base params:")
for k, v in strategy.base_params.items():
    logger.info("  %s: %s", k, v)

logger.info("Stock group: %s", stock_group)
logger.info("Time group: %s", timegroup)
logger.info("Optuna study name: %s", optuna_study_name)
logger.info("Optuna duration (seconds): %s", optuna_duration)
logger.info("Global start date: %s", global_start_date)
logger.info("Final evaluation date: %s", final_eval_date)
logger.info("Test days: %s", n_test_days)
logger.info("Number of splits: %s", n_splits)
logger.info("Number of startup trials: %s", n_startup_trials)
logger.info("Training days reserved: %s", n_training_days_reserved)
logger.info("Optimization direction: %s", direction)

if __name__ == "__main__":
    test_dates = [final_eval_date - datetime.timedelta(days=i) for i in range(n_test_days)][::-1]

    ls = LoadupSamples(
        train_start_date=global_start_date,
        test_dates=test_dates,
        treegroup=stock_group,
        timegroup=timegroup,
        params=loadup_params,
    )
    ls.load_samples()

    optuna_client = OptunaClient(
        ls=ls,
        n_splits=n_splits,
        n_test_days=n_test_days,
        n_training_days=n_training_days_reserved,
    )

    objective = optuna_client.make_objective(
        strategy=strategy, 
        preset_params=loadup_params
    )

    optuna_tuner = OptunaTuning(
        studyname=optuna_study_name,
        n_startup_trials=n_startup_trials,
        studytime=optuna_duration,
        objective=objective,
        direction=direction,
    )

    study, df = optuna_tuner.run()

    Path("outputs").mkdir(exist_ok=True)
    output_path = Path("outputs") / f"optuna_results_{stock_group_short}_{formatted_date}.parquet"
    df.to_parquet(output_path)
    logger.info("Saved Optuna results to %s", output_path)
    logger.info("Optuna study finished with %s completed trials.", len(df))

