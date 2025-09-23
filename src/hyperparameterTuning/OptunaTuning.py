import optuna
import pandas as pd
import polars as pl

import logging
logger = logging.getLogger(__name__)

class OptunaTuning():
    storage_name = "sqlite:///sandbox_optuna.db"

    def __init__(self,
            studyname: str,
            n_startup_trials: int,
            studytime: int,
            objective: callable[[optuna.Trial], float],
            direction: str,
        ):
        self.studyname = studyname
        self.n_startup_trials = n_startup_trials
        self.studytime = studytime
        self.objective = objective
        self.direction = direction

    def run(self) -> tuple[optuna.study.Study, pd.DataFrame]:
        logger.info(f"Starting Optuna study: {self.studyname}")
        optuna.logging.enable_propagation()
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials
        )
        study = optuna.create_study(
            study_name=self.studyname,
            storage=self.storage_name,
            direction=self.direction,
            load_if_exists=True,
            sampler=sampler,
        )
        study.optimize(
            self.objective,
            timeout=self.studytime
        )

        df = self.__analyze_study(study)

        return study, df

    def __analyze_study(self, study: optuna.study.Study) -> pd.DataFrame:
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best score: {study.best_value}")

        df: pd.DataFrame = study.trials_dataframe()
        logger.info("\nTrials DataFrame:")
        logger.info(df.sort_values("value").to_string())

        param_importances = optuna.importance.get_param_importances(study)
        logger.info("Parameter Importances:")
        for key, value in param_importances.items():
            logger.info(f"{key}: {value}")

        df_pl = pl.from_pandas(df)
        df_roll_mean = df_pl.sort("value").select(
            [pl.col("value")] + 
            [pl.col(c).rolling_mean(window_size=10).alias(f"{c}_rollmean10") for c in df.columns if c.startswith("params_")]
        )
        df_roll_min = df_pl.sort("value").select(
            [pl.col("value")] + 
            [pl.col(c).rolling_min(window_size=10).alias(f"{c}_rollmin10") for c in df.columns if c.startswith("params_")]
        )
        df_roll_max = df_pl.sort("value").select(
            [pl.col("value")] + 
            [pl.col(c).rolling_max(window_size=10).alias(f"{c}_rollmax10") for c in df.columns if c.startswith("params_")]
        )

        logger.info(df_roll_mean.to_pandas().to_string())
        logger.info(df_roll_min.to_pandas().to_string())
        logger.info(df_roll_max.to_pandas().to_string())

        return df_pl.sort("value").to_pandas()