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

from src.hyperparameterTuning.HelperFunctions import HelperFunctions

logger = logging.getLogger(__name__)


class StratClusteringAnomalies(BaseStrategy):
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
        "FilterSamples_cat_volatility_qup0.95": True,
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
            "t_win":                        trial.suggest_int("t_win", 3, 55, step=1),
            "n_clusters":                   trial.suggest_int("n_clusters", 4, 10, step=1),
            "min_n_targets":                50,
            "min_n_training_samples":       1000,
            "trcluster_mean_min_threshold": trial.suggest_float("trcluster_mean_min_threshold", 1.0, 1.01),
        })
        params.update({
            "n_max_anom_cluster":       trial.suggest_int("n_max_anom_cluster", 1, params["n_clusters"]-1, step=1),
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
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(ytr_tree, ytr_tree_low, ytr_tree_high, ytr_tree_open)
        def default_res():
            m_te = np.zeros(Xte_time.shape[0], dtype=bool)
            sl_te = sl_val * np.ones(Xte_time.shape[0], dtype=float)
            tp_te = tp_val * np.ones(Xte_time.shape[0], dtype=float)
            return m_te, sl_te, tp_te

        t_win = int(opt_params.get("t_win", 5))
        n_clusters = int(opt_params.get("n_clusters", 4))
        min_n_targets = int(opt_params.get("min_n_targets", 100))
        n_max_anom_cluster = int(opt_params.get("n_max_anom_cluster", 1))
        min_n_training_samples = int(opt_params.get("min_n_training_samples", 10000))
        trcluster_mean_min_threshold = opt_params.get("trcluster_mean_min_threshold", 0.6)

        ytr_opt = ytr_tree[:,-1]

        Xd_tr = Xtr_time[:, -t_win:, :].reshape(Xtr_time.shape[0], -1)
        Xd_te = Xte_time[:, -t_win:, :].reshape(Xte_time.shape[0], -1)

        if n_clusters >= Xd_tr.shape[0]:
            raise ValueError("n_clusters must be less than the number of training samples.")

        mask_tr = np.ones(Xd_tr.shape[0], dtype=bool)
        mask_te_sel = np.zeros(Xd_te.shape[0], dtype=bool)

        max_iter = 50
        it = 0
        while mask_te_sel.sum() < min_n_targets and it < max_iter:
            it += 1

            n_tr_left = mask_tr.sum()
            n_te_sel  = mask_te_sel.sum()
            if n_tr_left < min_n_training_samples:
                logger.info(" Not enough training samples left (%d < %d). Stopping.", n_tr_left, min_n_training_samples)
                break

            logger.info(f" Clustering iteration {it}: n_tr_left={n_tr_left}, n_te_sel={n_te_sel}")

            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init="auto",
                init="k-means++",
                random_state=self._random_state,
            )
            Xtr_curr = Xd_tr[mask_tr]
            Xte_pool = Xd_te[~mask_te_sel]

            lab_tr = kmeans.fit_predict(Xtr_curr)
            lab_te = kmeans.predict(Xte_pool)
            
            clusters = np.arange(n_clusters)
            n_samples_tr_per_cluster = np.array([(lab_tr == c).sum() for c in clusters])
            n_samples_te_per_cluster = np.array([(lab_te == c).sum() for c in clusters])

            ytr_opt_masked = ytr_opt[mask_tr]
            mean_tar_per_cluster = np.array([
                ytr_opt_masked[lab_tr == c].mean() if (lab_tr == c).any() else 1.0
                for c in clusters
            ])
            
            for c in clusters:
                logger.info(f"   Cluster {c}: n_tr_samples={n_samples_tr_per_cluster[c]}, n_te_samples={n_samples_te_per_cluster[c]}, mean_target={mean_tar_per_cluster[c]:.4f}")

            argsort_n_clusters = np.argsort(n_samples_tr_per_cluster)
            subt_mask_tr = np.zeros_like(mask_tr, dtype=bool)
            add_mask_te = np.zeros_like(mask_te_sel, dtype=bool)
            train_idx_current = np.where(mask_tr)[0]
            test_idx_current = np.where(~mask_te_sel)[0]
            for c_idx in argsort_n_clusters[:n_max_anom_cluster]:
                c = clusters[c_idx]
                n_c_tr = n_samples_tr_per_cluster[c_idx]
                n_c_te = n_samples_te_per_cluster[c_idx]
                mean_tar_c = mean_tar_per_cluster[c_idx]
                logger.debug(f"   Commencing cluster {c}")
                to_add_to_te = True
                if n_c_tr == 0:
                    logger.debug("      Cluster has no training samples.")
                    to_add_to_te = False
                if n_c_te == 0:
                    logger.debug("      Cluster has no test samples.")
                    to_add_to_te = False
                if mean_tar_c <= trcluster_mean_min_threshold:
                    logger.debug("      Mean target is below threshold.")
                    to_add_to_te = False

                idx_tr_masked = train_idx_current[lab_tr == c]
                idx_te_masked = test_idx_current[lab_te == c]
                if idx_tr_masked.size != 0: # always remove rare train samples
                    subt_mask_tr[idx_tr_masked] = True
                if to_add_to_te and idx_te_masked.size != 0:
                    add_mask_te[idx_te_masked] = True
            
            mask_tr &= ~subt_mask_tr
            mask_te_sel |= add_mask_te

        res_mask, sl_vec, tp_vec = default_res()
        res_mask = mask_te_sel

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
        sl_tr_vec = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te_vec = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr_vec = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te_vec = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        logger.info(
            "  Precompute -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(f"  Precompute -> sl {sl_val} | tp: {tp_val}")

        return mask_train, mask_test, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec