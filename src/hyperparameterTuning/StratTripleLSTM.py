"""Triple LSTM strategy for Optuna tuning."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import optuna
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels

from src.hyperparameterTuning.HelperFunctions import HelperFunctions
from src.hyperparameterTuning.HelperMetrics import HelperMetrics

logger = logging.getLogger(__name__)


class StratTripleLSTM(BaseStrategy):

    expected_load_params = {
        "idxAfterPrediction": 1,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "FilterSamples_cat_over20": True,
        "FilterSamples_cat_under2000": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.2": False,
        "FilterSamples_cat_volatility_qdown0.025": False,
        "FilterSamples_cat_volatility_qup0.8": True,
        "FilterSamples_cat_predictability_qup0.9": False,
    }

    base_params = {}

    def __init__(self, *, device: str | None = None, random_state: int | None = 0) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._random_state = random_state
        self._scaler_cls = StandardScaler
        logger.info("Initialized Triple LSTM on device %s", self._device)

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        params = dict(self.base_params)

        params.update({
            "val_split":                trial.suggest_float("val_split", 0.01, 0.1, log=True),
            "t_win":                    trial.suggest_int("t_win", 5, 70, step=5),
            "time_inc_factor":          trial.suggest_float("time_inc_factor", 1.0, 60.0),
            "LSTM_units":               16,
            "LSTM_num_layers":          1,
            "LSTM_learning_rate":       trial.suggest_float("LSTM_learning_rate", 2e-4, 1e-3, log=True),
            "LSTM_dropout":             0.05,
            "LSTM_inter_dropout":       0.05,
            "LSTM_recurrent_dropout":   0.05,
            "LSTM_epochs":              4,
            "LSTM_l1":                  0.001,
            "LSTM_l2":                  0.001,
            "LSTM_conv1d_kernel_size":  5,
            "min_n_tar":                25,
            
            "ytree_kind":               trial.suggest_categorical("ytree_kind", ["last", "abslast"]),
            "tp_buffer_pct":            trial.suggest_float("tp_buffer_pct", 0.01, 0.4, log=True),
            "sl_tau":                   trial.suggest_float("sl_tau", 0.1, 0.9),
            "tp_abovezero_thr":         trial.suggest_float("tp_abovezero_thr", 0.005, 0.05, log=True),
            "sl_preds_exp_factor":      trial.suggest_float("sl_preds_exp_factor", 0.01, 0.5, log=True),
            "tp_preds_exp_factor":      trial.suggest_float("tp_preds_exp_factor", 0.01, 0.5, log=True),
        })

        return params

    def run(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        ytr_tree_low: np.ndarray,
        ytr_tree_high: np.ndarray,
        ytr_tree_open: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        treenames: list[str] | None,
        timenames: list[str] | None,
        meta_train: pl.DataFrame | None,
        meta_test: pl.DataFrame | None,
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del Xtr_tree, Xte_tree, treenames, timenames, meta_train, meta_test  #unused

        if Xtr_time.ndim != 3 or Xte_time.ndim != 3:
            raise ValueError("Xtr_time and Xte_time must be 3-dimensional arrays.")
        
        t_win = int(opt_params.get("t_win", 5))
        time_factor = float(opt_params.get("time_inc_factor", 1.0))
        val_split = float(opt_params.get("val_split", 0.1))
        tp_buffer_pct = float(opt_params.get("tp_buffer_pct", 0.1))
        sl_tau = float(opt_params.get("sl_tau", 0.1))
        tp_abovezero_thr = float(opt_params.get("tp_abovezero_thr", 0.02))
        sl_preds_exp_factor = float(opt_params.get("sl_preds_exp_factor", 1.0))
        tp_preds_exp_factor = float(opt_params.get("tp_preds_exp_factor", 1.0))
        tp_close_thr = 0.005
        sl_max = 0.995
        tp_min = 1.005
        q_up =  1 - opt_params["min_n_tar"] / Xte_time.shape[0]

        
        if opt_params["ytree_kind"] == "last":
            ytr_tree_opt = ytr_tree[:, -1]
        elif opt_params["ytree_kind"] == "abslast":
            ytr_tree_opt = np.abs(ytr_tree[:, -1])
            
        def default_res():
            m_te = np.zeros(Xte_time.shape[0], dtype=bool)
            sl_te = 0.82 * np.ones(Xte_time.shape[0], dtype=float)
            tp_te = 1.07 * np.ones(Xte_time.shape[0], dtype=float)
            return m_te, sl_te, tp_te
        
        def to_time(x):
            return np.clip(np.tanh(np.log(np.clip(x, 1e-6, None)) * time_factor) / 2.0 + 0.5, 1e-6, 1 - 1e-6)
        def to_tree(y):
            return np.exp(np.arctanh((y - 0.5) * 2.0)/time_factor)
        
        # Get tp target for LSTM
        y_high_last = ytr_tree_high[:,-1]
        y_close_last = ytr_tree[:,-1]
        #logger.debug("  y_high_last stats: min %.4f | max %.4f | mean %.4f", np.min(y_high_last), np.max(y_high_last), np.mean(y_high_last))
        #logger.debug("  y_close_last stats: min %.4f | max %.4f | mean %.4f", np.min(y_close_last), np.max(y_close_last), np.mean(y_close_last))
        mask_tp_hit_close = y_high_last <= y_close_last + tp_close_thr
        tp_tar                    = y_high_last                    - tp_buffer_pct * (y_high_last - 1.0)
        tp_tar[mask_tp_hit_close] = y_high_last[mask_tp_hit_close] + tp_buffer_pct * (y_high_last[mask_tp_hit_close] - 1.0)
        tp_tar = np.maximum(tp_tar, tp_min)

        logger.debug("  tp_tar stats: min %.4f | max %.4f | mean %.4f", np.min(tp_tar), np.max(tp_tar), np.mean(tp_tar))
        logger.debug("    tp hit close stats: number %s, ratio %.4f", np.sum(mask_tp_hit_close), np.sum(mask_tp_hit_close) / mask_tp_hit_close.size)
        
        # Get sl target for LSTM 
        tp_mirror = 1.0/(tp_tar + 1e-7)
        y_low_last = ytr_tree_low[:,-1]
        #logger.debug("  y_low_last stats: min %.4f | max %.4f | mean %.4f", np.min(y_low_last), np.max(y_low_last), np.mean(y_low_last))
        mask_high_above = y_high_last >= tp_abovezero_thr + 1.0
        mask_high_below = y_high_last < tp_abovezero_thr + 1.0   
        sl_tar = np.zeros_like(y_low_last) 
        sl_tar[mask_high_below] = sl_max
        sl_tar[mask_high_above] = np.minimum(tp_mirror[mask_high_above] * (1-sl_tau) + sl_tau * y_low_last[mask_high_above], tp_mirror[mask_high_above])
        
        logger.debug("  sl_tar stats: min %.4f | max %.4f | mean %.4f", np.min(sl_tar), np.max(sl_tar), np.mean(sl_tar))
        logger.debug("    sl high above stats: number %s, ratio %.4f", np.sum(mask_high_above), np.sum(mask_high_above) / mask_high_above.size)

        ytr_time_opt = to_time(ytr_tree_opt)
        sl_tar_time = to_time(sl_tar)
        tp_tar_time = to_time(tp_tar)

        N, T, F = Xtr_time.shape
        Xdtr_time = Xtr_time[:, -t_win:, :]
        Xdte_time = Xte_time[:, -t_win:, :]

        mm = MachineModels(params=opt_params)
        try:
            val_split_n = max(1, int(N * (1-val_split)))
            model_close, info_close = mm.run_LSTM_torch(
                X_train=Xdtr_time[:val_split_n],
                y_train=ytr_time_opt[:val_split_n],
                X_test=Xdtr_time[val_split_n:],
                y_test=ytr_time_opt[val_split_n:],
                device=self._device,
                logger_disabled=True,
            )
            model_sl, info_sl = mm.run_LSTM_torch(
                X_train=Xdtr_time[:val_split_n],
                y_train=sl_tar_time[:val_split_n],
                X_test=Xdtr_time[val_split_n:],
                y_test=sl_tar_time[val_split_n:],
                device=self._device,
                logger_disabled=True,
            )
            model_tp, info_tp = mm.run_LSTM_torch(
                X_train=Xdtr_time[:val_split_n],
                y_train=tp_tar_time[:val_split_n],
                X_test=Xdtr_time[val_split_n:],
                y_test=tp_tar_time[val_split_n:],
                device=self._device,
                logger_disabled=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[LSTM] training failed: %s", exc)
            return default_res()

        val_rmse_close = float(info_close.get("val_rmse", float("inf")))
        val_rmse_sl = float(info_sl.get("val_rmse", float("inf")))
        val_rmse_tp = float(info_tp.get("val_rmse", float("inf")))
        
        logger.debug("  LSTM val RMSE -> close: %.6f | sl: %.6f | tp: %.6f", val_rmse_close, val_rmse_sl, val_rmse_tp)

        # close predictions
        preds_close = mm.predict_LSTM_torch(
            model_close,
            Xdte_time,
            device=self._device,
        )
        preds_close = np.asarray(preds_close)
        if preds_close.size == 0 or not np.all(np.isfinite(preds_close)):
            logger.warning("[LSTM] close predictions are empty or non-finite.")
            return default_res()

        logger.info("  close preds stats: min %.4f | max %.4f | mean %.4f | std %.4f", np.min(preds_close), np.max(preds_close), np.mean(preds_close), np.std(preds_close))

        # sl predictions
        preds_sl = mm.predict_LSTM_torch(
            model_sl,
            Xdte_time,
            device=self._device,
        )
        preds_sl = np.asarray(preds_sl)
        if preds_sl.size == 0 or not np.all(np.isfinite(preds_sl)):
            logger.warning("[LSTM] sl predictions are empty or non-finite.")
            return default_res()
        preds_sl_tree = to_tree(preds_sl)
        mean_pred_sl = np.mean(preds_sl_tree)
        
        logger.info("  sl preds stats: min %.4f | max %.4f | mean %.4f | std %.4f", np.min(preds_sl_tree), np.max(preds_sl_tree), np.mean(preds_sl_tree), np.std(preds_sl_tree))

        # tp predictions
        preds_tp = mm.predict_LSTM_torch(
            model_tp,
            Xdte_time,
            device=self._device,
        )
        preds_tp = np.asarray(preds_tp)
        if preds_tp.size == 0 or not np.all(np.isfinite(preds_tp)):
            logger.warning("[LSTM] tp predictions are empty or non-finite.")
            return default_res()
        preds_tp_tree = to_tree(preds_tp)
        mean_pred_tp = np.mean(preds_tp_tree)

        logger.info("  tp preds stats: min %.4f | max %.4f | mean %.4f | std %.4f", np.min(preds_tp_tree), np.max(preds_tp_tree), np.mean(preds_tp_tree), np.std(preds_tp_tree))

        thr = np.quantile(preds_close, q_up)
        selection_mask = preds_close >= thr
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree, 
            ytr_tree_low, 
            ytr_tree_high, 
            ytr_tree_open,
            n_grid = 15,
            spread_cost=0.0000,
            commission=0.0,
        )
        sl_rmse = np.sqrt(np.mean(((preds_sl_tree - mean_pred_sl)) ** 2))
        tp_rmse = np.sqrt(np.mean(((preds_tp_tree - mean_pred_tp)) ** 2))
        sl_te = sl_val * (1.0 + (preds_sl_tree - mean_pred_sl) / sl_rmse * 0.01 * sl_preds_exp_factor)
        tp_te = tp_val * (1.0 + (preds_tp_tree - mean_pred_tp) / tp_rmse * 0.01 * tp_preds_exp_factor)
        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        
        score_tr = np.random.rand(Xtr_tree.shape[0])
        score_te = np.random.rand(Xte_tree.shape[0])

        return mask_train, selection_mask, sl_tr, sl_te, tp_tr, tp_te, score_tr, score_te

    def precompute(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        ytr_tree_low: np.ndarray,
        ytr_tree_high: np.ndarray,
        ytr_tree_open: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        treenames: list[str] | None,
        timenames: list[str] | None,
        meta_train: pl.DataFrame | None,
        meta_test: pl.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute pre-selection masks using categorical filters."""
        _ = (Xtr_time, Xte_time, timenames)

        if treenames is None or meta_train is None or meta_test is None:
            raise ValueError("treenames, meta_train and meta_test are required.")

        params = self.precompute_params

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

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid = 5,
            spread_cost=0.0000,
            commission=0.0,
        )
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        logger.info(
            "  Precompute -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(f"  Precompute -> sl {sl_tr} | tp: {tp_tr}")

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te