import datetime
import logging
from pathlib import Path

from src.hyperparameterTuning.OptunaClient import OptunaClient
from src.hyperparameterTuning.OptunaTuning import OptunaTuning
from src.hyperparameterTuning.StratFilterSamples import StratFilterSamples
from src.predictionModule.LoadupSamples import LoadupSamples

import treetimeParams


stock_group = "group_finanTo2011"
stock_group_short = stock_group.replace("group_", "")

formatted_date = datetime.datetime.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f"logs/output_optuna_TreeTime_{stock_group_short}_{formatted_date}.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
logger = logging.getLogger(__name__)

params = treetimeParams.params
logger.info("Params: %s", params)


def main() -> None:
    optuna_study_name = f"Optuna_{stock_group_short}_{formatted_date}"
    optuna_duration = 60 * 60 * 10
    global_start_date = datetime.date(2014, 1, 1)
    final_eval_date = datetime.date(2025, 7, 15)
    test_horizon_days = 7
    n_splits = 200
    n_startup_trials = max(20, n_splits // 5)
    training_window_days = params.get("Treetime_LSTM_days_to_train", 365)

    test_dates = [final_eval_date - datetime.timedelta(days=i) for i in range(test_horizon_days)][::-1]

    ls = LoadupSamples(
        train_start_date=global_start_date,
        test_dates=test_dates,
        treegroup=stock_group,
        params=params,
    )
    ls.load_samples()

    optuna_client = OptunaClient(
        ls=ls,
        n_splits=n_splits,
        n_test_days=test_horizon_days,
        n_training_days=training_window_days,
    )

    strategy_name = params.get("TreeTime_FilterSamples_method", "lincomb")
    strategy = StratFilterSamples(filter_method=strategy_name)

    # Log the selected split dates for traceability
    try:
        optuna_client.get_split_dates(
            final_split_date=final_eval_date,
            start_train_date=global_start_date,
        )
    except ValueError as exc:
        logger.warning("Unable to sample split dates for logging: %s", exc)

    objective = optuna_client.make_objective(strategy=strategy)

    optuna_tuner = OptunaTuning(
        studyname=optuna_study_name,
        n_startup_trials=n_startup_trials,
        studytime=optuna_duration,
        objective=objective,
        direction="maximize",
    )

    study, df = optuna_tuner.run()

    Path("outputs").mkdir(exist_ok=True)
    output_path = Path("outputs") / f"optuna_results_{stock_group_short}_{formatted_date}.parquet"
    df.to_parquet(output_path)
    logger.info("Saved Optuna results to %s", output_path)
    logger.info("Optuna study finished with %s completed trials.", len(df))


if __name__ == "__main__":
    main()
