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

class StratCatSampling(BaseStrategy):
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
        #"FilterSamples_cat_highestShareholderEquity_q0.2": False,
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
        opt_params["min_n_tar"] = 200
        opt_params["highestShareholderEquity_q"] = trial.suggest_float("highestShareholderEquity_q", 0.98, 0.995)
        opt_params["volatility_q"] = trial.suggest_float("volatility_q", 0.04, 0.07)
        opt_params["predictability_q"] = trial.suggest_float("predictability_q", 0.98, 0.995)
        
        opt_params["volatility_w"] = trial.suggest_int("volatility_w", 40, 60)  
        opt_params["predictability_w"] = trial.suggest_int("predictability_w", 30, 50)

        # quantile selection respecting min_n_tar
        opt_params["volatility_dir"] = "qdown"
        opt_params["predictability_dir"] = "qup"

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
        
        return np.ones(Xte_tree.shape[0], dtype=bool), 0, 0  # Dummy implementation
        # TODO: FIX ME BELOW
        """ 
        min_n_tar = opt_params["min_n_tar"]
        
        volat_w = opt_params["volatility_w"]
        volat_dir = opt_params["volatility_dir"]
        volat_q = opt_params["volatility_q"]
        
        predic_w = opt_params["predictability_w"]
        predic_dir = opt_params["predictability_dir"]
        predic_q = opt_params["predictability_q"]
        
        highequity_q = opt_params["highestShareholderEquity_q"]
        
        key_volat = f"FilterSamples_cat_volatility_w{volat_w}_{volat_dir}{volat_q:.2f}"
        key_predic = f"FilterSamples_cat_predictability_w{predic_w}_{predic_dir}{predic_q:.2f}"
        key_highequity = f"FilterSamples_cat_highestShareholderEquity_q{highequity_q:.2f}"
        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
        
        params = dict(opt_params)
        params.update({
            key_volat: True,
            key_predic: True,
            key_highequity: True,
        })

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

        # Update masks in-place while preserving alignment
        mask_train[mask_train] &= cat_train
        if cat_test is not None:
            mask_test[mask_test] &= cat_test
            
        logger.debug(f"  Masks after categorical filtering -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%",)
        logger.debug(f"  Absolute values of masks -> train kept: {mask_train.sum()} | test kept: {mask_test.sum()}")

        score_train = fs.evaluate_mask(
            mask_train,
            meta_train["date"],
            ytr_tree[:, -1],
        )
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid = 20,
            spread_cost=0.0,
            commission=0.0,
        )
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        logger.info("  Score (train) = %s", score_train)

        return mask_test, sl_te, tp_te
        """

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