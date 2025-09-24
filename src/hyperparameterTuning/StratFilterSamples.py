import numpy as np
import optuna
import polars as pl
import datetime

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta

import logging
logger = logging.getLogger(__name__)

class StratFilterSamples(BaseStrategy):
    default_params = {
        "idxAfterPrediction": 5,
        "timesteps": 60,
        "target_option": "last",
        "LoadupSamples_time_scaling_stretch": True,
        "LoadupSamples_time_inc_factor": 61,
        "FilterSamples_q_up": 0.985,
        "FilterSamples_days_to_train_end": 115,
        "FilterSamples_cat_over20": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_lincomb_epochs": 5,
        "FilterSamples_lincomb_show_progress": False,
        "FilterSamples_lincomb_featureratio": 0.5,
        "FilterSamples_lincomb_itermax": 1,
        "FilterSamples_lincomb_init_toprand": 3,
        "FilterSamples_lincomb_batch_size": 2**12,
        "FilterSamples_taylor_horizon_days": 50,
        "FilterSamples_taylor_roll_window_days": 10,
        "FilterSamples_taylor_weight_slope": 1.268923,
    }

    def __init__(self, filter_method: str, base_params: dict = {}) -> None:
        self.base_params = {**self.default_params, **base_params}
        if filter_method not in {"lincomb", "taylor"}:
            raise ValueError("filter_method must be either 'lincomb' or 'taylor'.")
        self.filter_method = filter_method

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        lincomb_space = {
            "FilterSamples_days_to_train_end": ("int", 250, 350, {"step": 10}),
            "FilterSamples_lincomb_lr": ("float", 1e-4, 5e-3, {"log": False}),
            "FilterSamples_lincomb_epochs": ("int", 10, 200, {"step": 10}),
            "FilterSamples_lincomb_probs_noise_std": ("float", 0.1, 0.3, {"log": False}),
            "FilterSamples_lincomb_subsample_ratio": ("float", 0.2, 0.4, {}),
            "FilterSamples_lincomb_sharpness": ("float", 2.0, 5.0, {"log": False}),
            "FilterSamples_lincomb_init_toprand": ("int", 2, 15, {}),
        }
        taylor_space = {
            "FilterSamples_days_to_train_end": ("int", 19, 48, {"step": 1}),
            "FilterSamples_taylor_horizon_days": ("int", 24, 28, {"step": 2}),
            "FilterSamples_taylor_roll_window_days": ("int", 1, 6, {"step": 1}),
            "FilterSamples_taylor_weight_slope": ("float", 1.5, 3.5, {"log": False}),
        }

        params = dict(self.base_params)
        if self.filter_method == "lincomb":
            params.update(self._parse_params(trial, lincomb_space))
        else:
            params.update(self._parse_params(trial, taylor_space))
        return params

    def score(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        yte_tree,
        treenames,
        timenames,
        meta_train,
        meta_test,
        opt_params: dict,
    ) -> float:
        mm: MachineModels = MachineModels(opt_params)

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree,
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=yte_tree,
            meta_train=meta_train,
            meta_test=meta_test,
            params=opt_params,
        )

        if self.filter_method == "lincomb":
            mask_train, mask_test = fs.lincomb_masks()
        else:
            mask_train, mask_test = fs.taylor_feature_masks()

        score_train = fs.evaluate_mask(
            mask_train,
            meta_train["date"],
            ytr_tree,
        )
        score_test = fs.evaluate_mask(
            mask_test,
            meta_test["date"],
            yte_tree,
        )

        logger.info("  Score (train) = %s", score_train)
        logger.info("  Score (test)  = %s", score_test)

        return float(score_test)

    def mask_precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        yte_tree,
        treenames,
        timenames,
        meta_train,
        meta_test,
    ) -> tuple[np.ndarray, np.ndarray]:
        if treenames is None or meta_train is None or meta_test is None:
            raise ValueError("treenames, meta_train and meta_test are required.")

        cat_mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        cat_mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree,
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=yte_tree,
            meta_train=meta_train,
            meta_test=meta_test,
            params=self.base_params,
        )

        cat_train, cat_test = fs.categorical_masks()
        cat_mask_train &= cat_train
        if cat_test is not None:
            cat_mask_test &= cat_test

        if self.filter_method == "taylor":
            dates_tr = meta_train["date"].unique().sort()
            last_day = dates_tr[-1]
            n_max_days_to_consider = 200
            start_day = last_day - datetime.timedelta(days=n_max_days_to_consider)
            
            filtered_train_mask: pl.Series = (meta_train["date"] >= start_day) & (meta_train["date"] <= last_day)
            cat_mask_train &= filtered_train_mask.fill_null(False).to_numpy()

        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * cat_mask_train.mean(),
            100 * cat_mask_test.mean(),
        )

        return cat_mask_train, cat_mask_test

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_params(trial: optuna.Trial, space: dict) -> dict:
        out = {}
        for name, spec in space.items():
            kind, lo, hi, kw = spec
            suggest = trial.suggest_int if kind == "int" else trial.suggest_float
            out[name] = suggest(name.replace("FilterSamples_", ""), lo, hi, **kw)
        return out
