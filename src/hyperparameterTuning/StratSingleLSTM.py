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


class StratSingleLSTM(BaseStrategy):

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
        "FilterSamples_cat_highestShareholderEquity_q0.2": False,
        "FilterSamples_cat_volatility_qdown0.025": False,
        "FilterSamples_cat_volatility_qup0.8": False,
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
            "val_split_ratio":          trial.suggest_float("val_split_ratio", 0.005, 0.02, log=True),
            "t_win":                    60, #trial.suggest_int("t_win", 5, 70, step=5),
            "time_inc_factor":          25.0, #trial.suggest_float("time_inc_factor", 1.0, 60.0),
            "LSTM_units":               trial.suggest_int("LSTM_units", 2, 8, step=2),
            "LSTM_num_layers":          1,
            "LSTM_learning_rate":       trial.suggest_float("LSTM_learning_rate", 1e-2, 2e-1, log=True),
            "LSTM_dropout":             0.05,
            "LSTM_inter_dropout":       0.05,
            "LSTM_recurrent_dropout":   0.05,
            "LSTM_epochs":              trial.suggest_int("LSTM_epochs", 6, 16, step=2),
            "LSTM_l1":                  0.001,
            "LSTM_l2":                  0.001,
            "LSTM_conv1d_kernel_size":  trial.suggest_int("LSTM_conv1d_kernel_size", 5, 15),
            "min_n_tar":                5,
            
            "ytree_kind":               "abslast", #trial.suggest_categorical("ytree_kind", ["last", "abslast"]),
            "vol_window":               24, #trial.suggest_int("vol_window", 4, 40, step=2),
            "filter_vol_qup":           trial.suggest_float("filter_vol_qup", 0.6, 0.95),
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
        if Xtr_time.ndim != 3 or Xte_time.ndim != 3:
            raise ValueError("Xtr_time and Xte_time must be 3-dimensional arrays.")
        
        t_win = int(opt_params.get("t_win", 5))
        time_factor = float(opt_params.get("time_inc_factor", 1.0))
        val_split_ratio = float(opt_params.get("val_split_ratio", 0.1))
        filter_vol_qup = float(opt_params.get("filter_vol_qup", 0.8))
        lstm_qup =  1 - opt_params["min_n_tar"] / (Xte_time.shape[0] * (1-filter_vol_qup))
        vol_window = opt_params.get("vol_window", 20)

        loc_params = dict(opt_params)
        loc_params[f"FilterSamples_cat_volatility_w{vol_window}_qup{filter_vol_qup}"] = True

        logger.debug("  Shape of Xtr_time: %s, Xte_time: %s", Xtr_time.shape, Xte_time.shape)
        logger.debug("  Running Single LSTM with t_win=%d, time_factor=%.4f, filter_vol_qup=%.4f, lstm_qup=%.4f", t_win, time_factor, filter_vol_qup, lstm_qup)

        if loc_params["ytree_kind"] == "last":
            ytr_tree_opt = ytr_tree[:, -1]
        else:
            ytr_tree_opt = np.abs(ytr_tree[:, -1])
        
        def to_time(x):
            return np.clip(np.tanh(np.log(np.clip(x, 1e-6, None)) * time_factor) / 2.0 + 0.5, 1e-6, 1 - 1e-6)
        #def to_tree(y):
        #    return np.exp(np.arctanh((y - 0.5) * 2.0)/time_factor)

        ytr_time_opt = to_time(ytr_tree_opt)

        Xdtr_time = Xtr_time[:, -t_win:, :]
        Xdte_time = Xte_time[:, -t_win:, :]
        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree_opt,
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=None,
            meta_train=meta_train,
            meta_test=meta_test,
            params=loc_params,
        )
        cat_train, cat_test = fs.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test

        logger.debug("After filtering, train samples kept: %.2f%% | test samples kept: %.2f%%", mask_train.mean(), mask_test.mean())

        def default_res():
            m_te = mask_test
            sl_te = 0.88 * np.ones(Xte_time.shape[0], dtype=float)
            tp_te = 1.21 * np.ones(Xte_time.shape[0], dtype=float)
            return m_te, sl_te, tp_te

        mm = MachineModels(params=loc_params)
        try:
            N = np.sum(mask_train)
            val_split_n = max(1, int(N * (1-val_split_ratio)))
            model_close, info_close = mm.run_LSTM_torch(
                X_train=Xdtr_time[mask_train][:val_split_n],
                y_train=ytr_time_opt[mask_train][:val_split_n],
                X_test=Xdtr_time[mask_train][val_split_n:],
                y_test=ytr_time_opt[mask_train][val_split_n:],
                device=self._device,
                logger_disabled=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[LSTM] training failed: %s", exc)
            return default_res()

        val_rmse_close = float(info_close.get("val_rmse", float("inf")))

        logger.info("  LSTM val RMSE -> close: %.6f, close*2/time_factor: %.6f", val_rmse_close, val_rmse_close * 2.0 / time_factor)

        # close predictions
        preds_close = mm.predict_LSTM_torch(
            model_close,
            Xdte_time[mask_test],
            device=self._device,
        )
        preds_close = np.asarray(preds_close)
        if preds_close.size == 0 or not np.all(np.isfinite(preds_close)):
            logger.warning("[LSTM] close predictions are empty or non-finite.")
            return default_res()

        logger.info("  close preds stats: min %.4f | max %.4f | mean %.4f | std %.4f", np.min(preds_close), np.max(preds_close), np.mean(preds_close), np.std(preds_close))

        thr = np.quantile(preds_close, lstm_qup)
        selection_mask = preds_close >= thr
        mask_test[mask_test] = selection_mask
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid = 20,
            spread_cost=0.0000,
            commission=0.0,
        )
        sl_te = sl_val * np.ones(Xte_time.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_time.shape[0], dtype=float)

        return mask_test, sl_te, tp_te

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