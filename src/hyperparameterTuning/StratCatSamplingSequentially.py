import numpy as np
import optuna
import polars as pl
import datetime
import copy

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta

import logging
logger = logging.getLogger(__name__)

class StratCatSamplingSequentially(BaseStrategy):
    expected_load_params = {
        "idxAfterPrediction": 5,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "FilterSamples_cat_over10": False,
        "FilterSamples_cat_under5000": False,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
    }

    base_params = {        
        #"FilterSamples_cat_volatility_w20_qdown0.2": False,
        #"FilterSamples_cat_volatility_w20_qup0.8": False,
        #"FilterSamples_cat_predictability_w20_qup0.8": False,
        #"FilterSamples_cat_predictability_w20_qdown0.2": False,
    }

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {}
        opt_params["min_n_tar_daily"] = 30
        opt_params["volprice_q"]                    = trial.suggest_float("volprice_q", 0.05, 0.90)
        opt_params["volprice_w"]                    = trial.suggest_int("volprice_w", 5, 60)
        opt_params["volatility_w"]                  = trial.suggest_int("volatility_w", 5, 60)
        opt_params["volatility_dir"]                = trial.suggest_categorical("volatility_dir", ["qup", "qdown"])
        opt_params["predictability_w"]              = trial.suggest_int("predictability_w", 5, 60)
        opt_params["predictability_dir"]            = trial.suggest_categorical("predictability_dir", ["qup", "qdown"])

        # quantile selection respecting min_n_tar
        if opt_params["volatility_dir"] == "qup":
            opt_params["volatility_q"] = trial.suggest_float("volatility_q", 0.501, 0.90)
        else:
            opt_params["volatility_q"] = trial.suggest_float("volatility_q", 0.1, 0.499)

        if opt_params["predictability_dir"] == "qup":
            opt_params["predictability_q"] = trial.suggest_float("predictability_q", 0.501, 0.90)
        else:
            opt_params["predictability_q"] = trial.suggest_float("predictability_q", 0.1, 0.499)
            
        order_key = trial.suggest_categorical(
            "cat_order",
            [
                "XVP",  # highest volumeprice, volatility, predictability
                "XPV",
                "VXP",
                "VPX",
                "PXV",
                "PVX",
            ],
        )

        mapping = {
            "X": "volumeprice",
            "V": "volatility",
            "P": "predictability",
        }
        first, second, third = (mapping[c] for c in order_key)

        opt_params["first_cat"] = first
        opt_params["second_cat"] = second
        opt_params["third_cat"] = third

        params = dict(self.base_params)
        params.update(opt_params)
        return params

    def run(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        ytr_tree_low,
        ytr_tree_high,
        ytr_tree_open,
        Xte_tree,
        Xte_time,
        treenames,
        timenames,
        meta_train,
        meta_test,
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: 
        min_n_tar_daily = opt_params["min_n_tar_daily"]
        n_dates_test = meta_test.get_column("date").n_unique()
        min_n_tar = min_n_tar_daily * n_dates_test
        
        volat_w = opt_params["volatility_w"]
        volat_dir = opt_params["volatility_dir"]
        volat_q = opt_params["volatility_q"]
        
        predic_w = opt_params["predictability_w"]
        predic_dir = opt_params["predictability_dir"]
        predic_q = opt_params["predictability_q"]
        
        volprice_q = opt_params["volprice_q"]
        volprice_w = opt_params["volprice_w"]
        
        first_cat = opt_params["first_cat"]
        second_cat = opt_params["second_cat"]
        third_cat = opt_params["third_cat"]
        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
        
        def make_filter_params(
            category: str, 
            base_params: dict,
            mask_test_step: np.ndarray
            ) -> dict:
            """Build params dict with the appropriate FilterSamples flag
            while enforcing min_n_tar via q_limit."""
            params = copy.deepcopy(base_params)

            n_test = int(mask_test_step.sum())
            if n_test <= 0:
                # Degenerate: force max restriction (no room to enforce min_n_tar)
                q_limit = 1.0
            else:
                q_limit = min(min_n_tar / n_test, 1.0)

            if category == "volatility":
                if volat_dir == "qdown":
                    q = max(volat_q, q_limit)
                else:  # "qup"
                    q = min(volat_q, 1.0 - q_limit)

                key = f"FilterSamples_cat_volatility_w{volat_w}_{volat_dir}{q:.2f}"
                params[key] = True

            elif category == "predictability":
                if predic_dir == "qdown":
                    q = max(predic_q, q_limit)
                else:  # "qup"
                    q = min(predic_q, 1.0 - q_limit)

                key = f"FilterSamples_cat_predictability_w{predic_w}_{predic_dir}{q:.2f}"
                params[key] = True

            elif category == "volumeprice":
                q = min(volprice_q, 1.0 - q_limit)
                key = f"FilterSamples_cat_volumeprice_w{volprice_w}_q{q:.2f}"
                params[key] = True

            else:
                raise ValueError(f"Unknown category: {category}")
            
            logger.debug(f"  key: {key}")

            return params
        
        def apply_step(
            category: str,
            mask_train: np.ndarray,
            mask_test: np.ndarray,
            base_params: dict
        ):
            step_params = make_filter_params(category, base_params, mask_test)

            fs = FilterSamples(
                Xtree_train=Xtr_tree[mask_train],
                ytree_train=ytr_tree[mask_train][:, -1],
                treenames=treenames,
                Xtree_test=Xte_tree[mask_test],
                ytree_test=None,
                meta_train=meta_train.filter(mask_train),
                meta_test=meta_test.filter(mask_test),
                params=step_params,
            )

            cat_train, cat_test = fs.categorical_masks()

            # Update masks in-place while preserving alignment
            mask_train[mask_train] &= cat_train
            if cat_test is not None:
                mask_test[mask_test] &= cat_test

            return fs, mask_train, mask_test
        
        logger.debug("Applying first category filter: %s", first_cat)
        _, mask_train, mask_test = apply_step(first_cat, mask_train, mask_test, opt_params)
        logger.debug(f"  After first cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")
        
        logger.debug("Applying second category filter: %s", second_cat)
        _, mask_train, mask_test = apply_step(second_cat, mask_train, mask_test, opt_params)
        logger.debug(f"  After second cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")
        
        logger.debug("Applying third category filter: %s", third_cat)
        fs_final, mask_train, mask_test = apply_step(third_cat, mask_train, mask_test, opt_params)
        logger.debug(f"  After third cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")
        
        score_train = fs_final.evaluate_mask(
            mask_train,
            meta_train["date"],
            ytr_tree[:, -1],
        )
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid=20,
            spread_cost=0.0,
            commission=0.0,
        )
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)
        
        score_tr = np.random.rand(Xtr_tree.shape[0])
        score_te = np.random.rand(Xte_tree.shape[0])

        logger.info("  Score (train) = %s", score_train)

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te, score_tr, score_te

    def precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        ytr_tree_low,
        ytr_tree_high,
        ytr_tree_open,
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
            ytree_train=ytr_tree[:,-1],
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

        sl_val, tp_val = 0.88, 1.1
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)
        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(f"  Precompute -> sl {sl_val} | tp: {tp_val}")

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te