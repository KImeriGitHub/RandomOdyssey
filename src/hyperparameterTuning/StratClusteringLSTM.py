"""Clustering-based LSTM strategy for Optuna tuning."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import optuna
import polars as pl
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels

from src.hyperparameterTuning.HelperFunctions import HelperFunctions
from src.hyperparameterTuning.HelperMetrics import HelperMetrics

logger = logging.getLogger(__name__)


class StratClusteringLSTM(BaseStrategy):
    """Strategy combining K-Means clustering with an LSTM predictor.

    The strategy clusters training samples using flattened temporal windows and
    trains one LSTM per cluster. The cluster that achieves the lowest validation
    RMSE is used to score the corresponding test members. Test samples are then
    filtered using a quantile threshold on the LSTM predictions and evaluated
    through the geometric mean of the selected tree targets.
    """

    expected_load_params = {
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "FilterSamples_cat_over20": True,
        "FilterSamples_cat_under2000": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.2": True,
        "FilterSamples_cat_volatility_qdown0.025": False,
        "FilterSamples_cat_volatility_qup0.8": True,
        "FilterSamples_cat_predictability_qup0.9": False,
    }

    base_params = {}

    def __init__(self, *, device: str | None = None, random_state: int | None = 0) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._random_state = random_state
        self._scaler_cls = StandardScaler
        logger.info("Initialized StratClusteringLSTM on device %s", self._device)

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        """Sample the search space for the clustering LSTM strategy."""
        params = dict(self.base_params)

        params.update({
            "val_split":                0.1,
            "t_win":                    trial.suggest_int("t_win", 40, 70, step=5),
            "time_inc_factor":          60,
            "n_clusters":               trial.suggest_int("n_clusters", 7, 10, step=1),
            "LSTM_units":               16,
            "LSTM_num_layers":          1,
            "LSTM_learning_rate":       trial.suggest_float("LSTM_learning_rate", 2e-4, 1e-3, log=True),
            "LSTM_dropout":             0.05,
            "LSTM_inter_dropout":       0.05,
            "LSTM_recurrent_dropout":   0.05,
            "LSTM_epochs":              30,
            "LSTM_l1":                  0.001,
            "LSTM_l2":                  0.001,
            "LSTM_conv1d_kernel_size":  5,
            "selection_quantile":       0.98, #trial.suggest_float("selection_quantile", 0.9, 0.99, log=True),
            "min_cluster_train":        500,
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
        """Train the clustering LSTM on the provided split and return a score."""
        del Xtr_tree, Xte_tree, treenames, timenames, meta_train, meta_test  #unused

        if Xtr_time.ndim != 3 or Xte_time.ndim != 3:
            raise ValueError("Xtr_time and Xte_time must be 3-dimensional arrays.")
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(ytr_tree, ytr_tree_low, ytr_tree_high, ytr_tree_open)
        ytr_tree_opt = np.max(ytr_tree, axis=1)
        def default_res():
            m_te = np.zeros(Xte_time.shape[0], dtype=bool)
            sl_te = sl_val * np.ones(Xte_time.shape[0], dtype=float)
            tp_te = tp_val * np.ones(Xte_time.shape[0], dtype=float)
            return m_te, sl_te, tp_te

        t_win = int(opt_params.get("t_win", 5))
        time_factor = opt_params.get("time_inc_factor")
        n_clusters = int(opt_params.get("n_clusters", 4))
        quantile_val = float(opt_params.get("selection_quantile", 0.95))
        min_cluster_train = int(opt_params.get("min_cluster_train", 50))

        ytr_time_scaled = np.tanh((ytr_tree_opt - 1.0) * time_factor) / 2.0 + 0.5

        Xd_tr = Xtr_time[:, -t_win:, :].reshape(Xtr_time.shape[0], -1)
        Xd_te = Xte_time[:, -t_win:, :].reshape(Xte_time.shape[0], -1)

        if n_clusters >= Xd_tr.shape[0]:
            logger.warning(
                "[LSTM] number of clusters %s is not smaller than training samples %s.",
                n_clusters,
                Xd_tr.shape[0],
            )
            return default_res()

        n_feat = Xtr_time.shape[-1]
        Xseq_tr = Xd_tr.reshape(-1, t_win, n_feat)
        Xseq_te = Xd_te.reshape(-1, t_win, n_feat)

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            init="k-means++",
            random_state=self._random_state,
        )
        lab_tr = kmeans.fit_predict(Xd_tr)
        lab_te = kmeans.predict(Xd_te)

        mm = MachineModels(params=opt_params)

        best_c = None
        best_model = None
        best_rmse = float("inf")

        for c in range(n_clusters):
            mask_tr = lab_tr == c
            mask_te = lab_te == c
            n_train = int(mask_tr.sum())
            n_test = int(mask_te.sum())
            if n_train < min_cluster_train:
                logger.info(
                    "[LSTM] cluster %s skipped (train size %s < %s): ytr_mean=%.4f",
                    c,
                    n_train,
                    min_cluster_train,
                    np.mean(ytr_tree_opt[mask_tr]) if mask_tr.sum() > 0 else float("nan"),
                )
                continue

            Xc_tr = Xseq_tr[mask_tr]
            yc_tr = ytr_time_scaled[mask_tr]
            if Xc_tr.size == 0 or yc_tr.size == 0:
                logger.info("[LSTM] cluster %s skipped due to empty data.", c)
                continue

            try:
                val_split_n = max(1, int(Xc_tr.shape[0] * (1-float(opt_params.get("val_split", 0.1)))))
                model_c, info = mm.run_LSTM_torch(
                    X_train=Xc_tr[:val_split_n],
                    y_train=yc_tr[:val_split_n],
                    X_test=Xc_tr[val_split_n:],
                    y_test=yc_tr[val_split_n:],
                    device=self._device,
                    logger_disabled=True,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[LSTM] cluster %s training failed: %s", c, exc)
                continue

            val_rmse = float(info.get("val_rmse", float("inf")))
            logger.info(
                "[LSTM] cluster %s: train=%s, test=%s, val_rmse=%.6f",
                c,
                n_train,
                n_test,
                val_rmse,
            )

            if np.isfinite(val_rmse) and val_rmse < best_rmse:
                best_c = c
                best_model = model_c
                best_rmse = val_rmse

        if best_c is None or best_model is None:
            logger.warning("[LSTM] no cluster produced a valid model.")
            return default_res()

        mask_te_best = lab_te == best_c
        if not np.any(mask_te_best):
            logger.warning("[LSTM] best cluster %s has no test members.", best_c)
            return default_res()

        preds = mm.predict_LSTM_torch(
            best_model,
            Xseq_te[mask_te_best],
            device=self._device,
        )
        preds = np.asarray(preds)
        if preds.size == 0 or not np.all(np.isfinite(preds)):
            logger.warning("[LSTM] predictions are empty or non-finite.")
            return default_res()

        thr = float(np.quantile(preds, quantile_val))
        selection_mask = preds >= thr

        res_mask, sl_vec, tp_vec = default_res()
        res_mask[mask_te_best] = selection_mask

        return res_mask, sl_vec, tp_vec

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
            ytree_train=np.max(ytr_tree, axis=1),
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

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(ytr_tree, ytr_tree_low, ytr_tree_high, ytr_tree_open)
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