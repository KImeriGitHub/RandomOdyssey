import logging
from typing import Callable, Tuple

import optuna
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class OptunaTuning:
    """Run and analyse an Optuna study."""

    def __init__(
        self,
        studyname: str,
        n_startup_trials: int,
        studytime: int,
        objective: Callable[[optuna.Trial], float],
        direction: str = "maximize",
        storage_name: str = "sqlite:///sandbox_optuna.db",
    ) -> None:
        self.studyname = studyname
        self.n_startup_trials = n_startup_trials
        self.studytime = studytime
        self.objective = objective
        self.direction = direction
        self.storage_name = storage_name

    def run(self) -> Tuple[optuna.study.Study, pd.DataFrame]:
        logger.info("Starting Optuna study: %s", self.studyname)
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

        sampler = optuna.samplers.TPESampler(n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(
            study_name=self.studyname,
            storage=self.storage_name,
            direction=self.direction,
            load_if_exists=True,
            sampler=sampler,
        )
        study.optimize(self.objective, timeout=self.studytime)

        df = self._analyze_study(study)
        return study, df

    def _analyze_study(self, study: optuna.study.Study) -> pd.DataFrame:
        logger.info("Best parameters: %s", study.best_params)
        logger.info("Best score: %s", study.best_value)

        df: pd.DataFrame = study.trials_dataframe()
        logger.info("\nTrials DataFrame:\n%s", df.sort_values("value").to_string())

        try:
            param_importances = optuna.importance.get_param_importances(study)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to compute parameter importances: %s", exc)
            param_importances = {}

        if param_importances:
            logger.info("Parameter Importances:")
            for key, value in param_importances.items():
                logger.info("%s: %s", key, value)

        df_pl = pl.from_pandas(df)
        param_cols = [c for c in df.columns if c.startswith("params_") and df_pl.schema[c] in pl.NUMERIC_DTYPES]
        logger.info("Parameter columns: %s", param_cols)
        logger.info("Non-Numeric parameter columns: %s", [c for c in df.columns if c.startswith("params_") and df_pl.schema[c] not in pl.NUMERIC_DTYPES])
        if param_cols:
            df_roll_mean = df_pl.sort("value").select(
                [pl.col("value")] + [pl.col(c).rolling_mean(window_size=10).alias(f"{c}_rollmean10") for c in param_cols]
            )
            df_roll_min = df_pl.sort("value").select(
                [pl.col("value")] + [pl.col(c).rolling_min(window_size=10).alias(f"{c}_rollmin10") for c in param_cols]
            )
            df_roll_max = df_pl.sort("value").select(
                [pl.col("value")] + [pl.col(c).rolling_max(window_size=10).alias(f"{c}_rollmax10") for c in param_cols]
            )

            logger.info(df_roll_mean.to_pandas().to_string())
            logger.info(df_roll_min.to_pandas().to_string())
            logger.info(df_roll_max.to_pandas().to_string())

        return df_pl.sort("value").to_pandas()
