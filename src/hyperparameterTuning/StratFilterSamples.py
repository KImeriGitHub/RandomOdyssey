import numpy as np
import optuna
import polars as pl
import datetime

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta

import logging
logger = logging.getLogger(__name__)

class StratFilterSamples(BaseStrategy):
    expected_load_params = {
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "FilterSamples_q_up": 0.975,
        "FilterSamples_cat_over20": True,
        "FilterSamples_cat_under2000": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.6": True,
        "FilterSamples_cat_volatility_qdown0.025": False,
        "FilterSamples_cat_volatility_qup0.975": False,
        "FilterSamples_cat_predictability_qup0.9": False,
    }

    base_params = {
        "FilterSamples_lincomb_probs_noise_std": 0.001,
        "FilterSamples_lincomb_subsample_ratio": 1.0,
        "FilterSamples_lincomb_sharpness": 1.0,
        "FilterSamples_lincomb_init_toprand": 3,
        "FilterSamples_lincomb_featureratio": 0.8,
        "FilterSamples_lincomb_itermax": 1,
        
        "FilterSamples_days_to_train_end": 300,
        "FilterSamples_taylor_horizon_days": 105,
        "FilterSamples_taylor_roll_window_days": 105,
        "FilterSamples_taylor_weight_slope": 0.0,
    }

    def __init__(self, filter_method: str) -> None:
        if filter_method not in {"lincomb", "taylor"}:
            raise ValueError("filter_method must be either 'lincomb' or 'taylor'.")
        self.filter_method = filter_method
        logger.info("Initialized StratFilterSamples with filter_method: %s", self.filter_method)

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        lincomb_space = {
            "FilterSamples_days_to_train_end": ("int", 200, 500, {"step": 100}),
            "FilterSamples_lincomb_lr": ("float", 5e-6, 1e-1, {"log": False}),
            "FilterSamples_lincomb_epochs": ("int", 1, 11, {"step": 2}),
            #"FilterSamples_lincomb_probs_noise_std": ("float", 0.01, 0.025, {"log": True}),
            #"FilterSamples_lincomb_subsample_ratio": ("float", 0.1, 0.8, {}),
            #"FilterSamples_lincomb_sharpness": ("float", 0.3, 1.5, {}),
            #"FilterSamples_lincomb_init_toprand": ("int", 1, 4, {}),
            #"FilterSamples_lincomb_featureratio": ("float", 0.15, 0.9, {}),
            #"FilterSamples_lincomb_itermax": ("int", 1, 3, {}),
        }
        taylor_space = {
            #"FilterSamples_days_to_train_end": ("int", 200, 500, {"step": 100}),
            #"FilterSamples_taylor_horizon_days": ("int", 2, 8, {"step": 1}),
            "FilterSamples_taylor_roll_window_days": ("int", 200, 400, {"step": 100}),
            #"FilterSamples_taylor_weight_slope": ("float", 0.1, 4.5, {"log": True}),
        }

        params = dict(self.base_params)
        if self.filter_method == "lincomb":
            params.update(self._parse_params(trial, lincomb_space))
        else:
            params.update(self._parse_params(trial, taylor_space))
        return params

    def run(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        treenames,
        timenames,
        meta_train,
        meta_test,
        opt_params: dict,
    ) -> float:
        mm: MachineModels = MachineModels(opt_params)

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree[:, -1],
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=None,
            meta_train=meta_train,
            meta_test=meta_test,
            params=opt_params,
        )

        sl_val, tp_val = HelperFunctions.optimal_sl_tp(ytr_tree)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        if self.filter_method == "lincomb":
            mask_train, mask_test = fs.lincomb_masks()
        else:
            mask_train, mask_test = fs.taylor_feature_masks()

        score_train = fs.evaluate_mask(
            mask_train,
            meta_train["date"],
            ytr_tree,
        )

        logger.info("  Score (train) = %s", score_train)

        return mask_test, sl_te, tp_te

    def precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        treenames,
        timenames,
        meta_train,
        meta_test,
    ) -> tuple[np.ndarray, np.ndarray]:
        if treenames is None or meta_train is None or meta_test is None:
            raise ValueError("treenames, meta_train and meta_test are required.")

        params = dict(self.precompute_params)

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree[:, -1],
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=None,
            meta_train=meta_train,
            meta_test=meta_test,
            params=params,
        )

        cat_train, cat_test = fs.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test

        if self.filter_method == "taylor":
            dates_tr = meta_train["date"].unique().sort()
            last_day = dates_tr[-1]
            n_max_days_to_consider = 200
            start_day = last_day - datetime.timedelta(days=n_max_days_to_consider)
            
            filtered_train_mask: pl.Series = (meta_train["date"] >= start_day) & (meta_train["date"] <= last_day)
            mask_train &= filtered_train_mask.fill_null(False).to_numpy()

        sl_val, tp_val = HelperFunctions.optimal_sl_tp(ytr_tree)
        sl_tr_vec = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te_vec = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr_vec = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te_vec = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(
            "  Precompute -> sl %.2f%% | tp: %.2f%%",
            sl_val,
            tp_val,
        )

        return mask_train, mask_test, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec

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
