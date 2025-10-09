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
        "FilterSamples_q_up": 0.6,
        "FilterSamples_cat_over20": True,
        "FilterSamples_cat_under2000": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.8": True
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
            "t_win":                    trial.suggest_int("t_win", 15, 55, step=5),
            "time_inc_factor":          60,
            "n_clusters":               trial.suggest_int("n_clusters", 3, 8, step=1),
            "LSTM_units":               16,
            "LSTM_num_layers":          1,
            "LSTM_learning_rate":       trial.suggest_float("LSTM_learning_rate", 1e-5, 1e-2, log=True),
            "LSTM_dropout":             0.05,
            "LSTM_inter_dropout":       0.05,
            "LSTM_recurrent_dropout":   0.05,
            "LSTM_epochs":              2,
            "LSTM_l1":                  0.001,
            "LSTM_l2":                  0.001,
            "LSTM_conv1d_kernel_size":  5,
            "selection_quantile":       trial.suggest_float("selection_quantile", 0.9, 0.99),
            "min_cluster_train":        500,
        })

        return params

    def score(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        yte_tree: np.ndarray,
        treenames: list[str] | None,
        timenames: list[str] | None,
        meta_train: pl.DataFrame | None,
        meta_test: pl.DataFrame | None,
        opt_params: dict,
    ) -> float:
        """Train the clustering LSTM on the provided split and return a score."""
        del Xtr_tree, Xte_tree, treenames, timenames, meta_train, meta_test  #unused

        if Xtr_time.ndim != 3 or Xte_time.ndim != 3:
            raise ValueError("Xtr_time and Xte_time must be 3-dimensional arrays.")

        t_win = int(opt_params.get("t_win", self.base_params["timesteps"]))
        time_factor = opt_params.get("time_inc_factor")
        n_clusters = int(opt_params.get("n_clusters", 4))
        quantile_val = float(opt_params.get("selection_quantile", 0.95))
        min_cluster_train = int(opt_params.get("min_cluster_train", 50))

        ytr_time_scaled = np.tanh((ytr_tree - 1.0) * time_factor) / 2.0 + 0.5

        Xd_tr = Xtr_time[:, -t_win:, :].reshape(Xtr_time.shape[0], -1)
        Xd_te = Xte_time[:, -t_win:, :].reshape(Xte_time.shape[0], -1)

        if n_clusters >= Xd_tr.shape[0]:
            logger.warning(
                "[LSTM] number of clusters %s is not smaller than training samples %s.",
                n_clusters,
                Xd_tr.shape[0],
            )
            return 1.0

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
                    "[LSTM] cluster %s skipped (train size %s < %s)",
                    c,
                    n_train,
                    min_cluster_train,
                )
                continue

            Xc_tr = Xseq_tr[mask_tr]
            yc_tr = ytr_time_scaled[mask_tr]
            if Xc_tr.size == 0 or yc_tr.size == 0:
                logger.info("[LSTM] cluster %s skipped due to empty data.", c)
                continue

            try:
                model_c, info = mm.run_LSTM_torch(
                    X_train=Xc_tr,
                    y_train=yc_tr,
                    X_test=None,
                    y_test=None,
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
            return 1.0

        mask_te_best = lab_te == best_c
        if not np.any(mask_te_best):
            logger.warning("[LSTM] best cluster %s has no test members.", best_c)
            return 1.0

        preds = mm.predict_LSTM_torch(
            best_model,
            Xseq_te[mask_te_best],
            device=self._device,
        )
        preds = np.asarray(preds)
        if preds.size == 0 or not np.all(np.isfinite(preds)):
            logger.warning("[LSTM] predictions are empty or non-finite.")
            return 1.0

        thr = float(np.quantile(preds, quantile_val))
        selection_mask = preds >= thr
        y_selected = yte_tree[mask_te_best][selection_mask]
        y_selected = y_selected[np.isfinite(y_selected)]

        if y_selected.size == 0:
            logger.warning("[LSTM] no test values selected after thresholding.")
            return 1.0

        score = self._geometric_mean_safe(y_selected)
        logger.info(
            "[LSTM] t_win=%s, clusters=%s -> best_cluster=%s, val_rmse=%.6f, "
            "selected=%s/%s, quantile=%.3f, score=%.6f",
            t_win,
            n_clusters,
            best_c,
            best_rmse,
            y_selected.size,
            mask_te_best.sum(),
            quantile_val,
            score,
        )

        if not np.isfinite(score) or score <= 0:
            return 1.0

        return float(score)

    def mask_precompute(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        yte_tree: np.ndarray,
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
            ytree_train=ytr_tree,
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=yte_tree,
            meta_train=meta_train,
            meta_test=meta_test,
            params=params,
        )
        cat_train, cat_test = fs.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test

        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )

        return mask_train, mask_test

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geometric_mean_safe(arr: Iterable[float]) -> float:
        """Compute a numerically stable geometric mean."""
        arr = np.asarray(list(arr), dtype=float)
        if arr.size == 0:
            return float("nan")
        minv = np.min(arr)
        shift = -minv + 1e-9 if minv <= 0 else 0.0
        return float(np.exp(np.mean(np.log(arr + shift))))

    @staticmethod
    def _resolve_feature_indices(category: str | Sequence[int] | None, n_features: int) -> list[int]:
        """Translate feature selection descriptors into explicit indices."""
        if category is None:
            return list(range(n_features))
        if isinstance(category, Sequence) and not isinstance(category, str):
            indices = [int(idx) for idx in category]
            if not indices:
                raise ValueError("feature indices cannot be empty.")
            valid = [idx for idx in indices if 0 <= idx < n_features]
            if not valid:
                raise ValueError("feature indices are out of range.")
            return valid

        if category == "first":
            return [0]
        if category == "second":
            return [1] if n_features > 1 else [0]
        if category == "first_two":
            return [idx for idx in range(min(2, n_features))]
        if category == "all":
            return list(range(n_features))

        raise ValueError(f"Unknown feature category: {category}")