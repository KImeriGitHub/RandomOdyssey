import numpy as np
import optuna
import polars as pl

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

import logging
logger = logging.getLogger(__name__)

class StratOHLCVFiltering(BaseStrategy):
    expected_load_params = {
        "idxAfterPrediction": 5,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
    }

    base_params = {
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

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        qup_high = opt_params["adjhigh_qup"]
        qdown_high = opt_params["adjhigh_qdown"]
        idxdays_high = opt_params["adjhigh_idxdays"]
        decay_high = opt_params["adjhigh_decay"]

        qup_low = opt_params["adjlow_qup"]
        qdown_low = opt_params["adjlow_qdown"]
        idxdays_low = opt_params["adjlow_idxdays"]
        decay_low = opt_params["adjlow_decay"]
        
        lstm_idx_adjhigh = timenames.index("FeatureLSTM_AdjHigh")
        lstm_idx_adjlow = timenames.index("FeatureLSTM_AdjLow")

        Xtr_time_adjhigh_window = Xtr_time[:, -idxdays_high:, lstm_idx_adjhigh]
        Xtr_time_adjlow_window = Xtr_time[:, -idxdays_low:, lstm_idx_adjlow]

        #Weights for time decay
        steps = np.arange(idxdays_high) # 0, 1, 2, ...
        weights_high = np.exp(-decay_high * (idxdays_high - 1 - steps))
        weights_high = weights_high / weights_high.sum()
        Xtr_time_adjhigh_vec = np.sum(Xtr_time_adjhigh_window * weights_high, axis=1)

        steps = np.arange(idxdays_low)
        weights_low = np.exp(-decay_low * (idxdays_low - 1 - steps))
        weights_low = weights_low / weights_low.sum()
        Xtr_time_adjlow_vec = np.sum(Xtr_time_adjlow_window * weights_low, axis=1)

        # Quantiles
        qup_high_val   = np.quantile(Xtr_time_adjhigh_vec, qup_high)
        qdown_high_val = np.quantile(Xtr_time_adjhigh_vec, qdown_high)
        mask_qup_high   = Xtr_time_adjhigh_vec > qup_high_val
        mask_qdown_high = Xtr_time_adjhigh_vec < qdown_high_val

        qup_low_val   = np.quantile(Xtr_time_adjlow_vec, qup_low)
        qdown_low_val = np.quantile(Xtr_time_adjlow_vec, qdown_low)
        mask_qup_low   = Xtr_time_adjlow_vec > qup_low_val
        mask_qdown_low = Xtr_time_adjlow_vec < qdown_low_val

        # Final mask
        mask_high = np.logical_and(mask_qup_high, mask_qdown_high)
        mask_low = np.logical_and(mask_qup_low, mask_qdown_low)
        
        mask_tar = np.logical_and(mask_train, np.logical_or(mask_high, mask_low))

        n_selected = np.count_nonzero(mask_tar)
        if n_selected < min_n_tar:
            remaining_idx = np.where(np.logical_and(mask_train, ~mask_tar))[0]
            n_to_add = min(min_n_tar - n_selected, remaining_idx.shape[0])
            if n_to_add > 0:
                extra_idx = np.random.choice(remaining_idx, size=n_to_add, replace=False)
                mask_tar[extra_idx] = True

        mask_train = mask_tar

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid=20,
            spread_cost=0.0,
            commission=0.0,
        )
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        return mask_test, sl_te, tp_te

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

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree, 
            ytr_tree_low, 
            ytr_tree_high, 
            ytr_tree_open,
            n_grid=7,
            spread_cost=0.0,
            commission=0.0,
        )
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