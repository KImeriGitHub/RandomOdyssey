from typing import Iterable, Union
import numpy as np
import polars as pl
import lightgbm as lgb

from src.hyperparameterTuning.HelperMetrics import HelperMetrics
from sklearn.feature_selection import r_regression

import logging
logger = logging.getLogger(__name__)

class HelperFunctions:
    def __init__():
        pass

    @staticmethod
    def top_leaf_labels_per_tree(
        model: lgb.Booster,
        X,
        y,
        tree_n_max: int,
        top_n_max: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For each tree t in [0, tree_idx_max], compute your score per leaf:
            score = (gmean_y - 1.96 * std_y / sqrt(count_y)) - 1.0  if count_y > 2
                    1.0 - 1.0 (=0)                                  otherwise
        Then pick the top `top_n_max` labels by score.
        Returns: int32 array of shape (top_n_max, tree_idx_max) with label ids; pads with -1.
        And score values associated with those labels, shape (top_n_max, tree_idx_max); pads with metric(1.0).
        Assumes y > 0 (for geometric mean).
        """
        leaf_mat = model.predict(X, pred_leaf=True)  # shape: (n_samples, n_trees_total)
        leaf_mat = leaf_mat.reshape(-1, 1) if leaf_mat.ndim == 1 else leaf_mat  # shape (n_samples, n_trees)
        n_trees = min(tree_n_max, leaf_mat.shape[1])

        y_arr = np.asarray(y, dtype=float)
        out_label = np.full((top_n_max, n_trees), -1, dtype=np.int32)
        out_score = np.full((top_n_max, n_trees), 1.0, dtype=float)

        for t in range(n_trees):
            labels_t = leaf_mat[:, t].astype(np.int32)

            df = pl.DataFrame({"label": labels_t, "y": y_arr})
            gb = df.group_by("label").agg([
                pl.count("y").alias("count_y"),
                pl.mean("y").alias("mean_y"),
                #pl.col("y").log().mean().exp().alias("gmean_y"),
                #pl.sum("y").alias("sum_y"),
                #pl.var("y").alias("var_y"),
                pl.std("y").alias("std_y"),
                #pl.max("y").alias("max_y"),
                #pl.min("y").alias("min_y"),
                #pl.median("y").alias("median_y"),
                #pl.quantile("y", 0.1).alias("q10_y"),
                #pl.quantile("y", 0.25).alias("q25_y"),
                #pl.quantile("y", 0.75).alias("q75_y"),
                #pl.quantile("y", 0.9).alias("q90_y"),
            ]).with_columns([
                    (pl.when(pl.col("count_y") > 2)
                        .then(pl.col("mean_y") - 1.96 * pl.col("std_y") / pl.col("count_y").sqrt())
                        .otherwise(1.0)
                    ).alias("_tmp")
            ]).with_columns([
                (pl.col("_tmp") - 1.0).alias("score")
            ])

            sorted_gb = gb.sort(["score", "count_y"], descending=[True, True]).select(["label", "score"])
            arr = sorted_gb.to_numpy()  # shape (n_labels, 2)
            k = min(top_n_max, arr.shape[0])
            if k > 0:
                out_label[:k, t] = arr[:k, 0].astype(np.int32)
                out_score[:k, t] = arr[:k, 1].astype(float)

        return out_label, out_score
    
    @staticmethod
    def exponential_moving_average(
        values: np.ndarray,
        window: int,
        alpha: float,
    ) -> np.ndarray:
        """
        values: (n_samples, n_timesteps)
        alpha: scalar EMA decay (0 < alpha <= 1)
        """
        values = np.asarray(values, dtype=np.float64)
        n, T = values.shape
        out = np.empty((n, T), dtype=np.float64)
        decay = 1.0 - alpha
        for t in range(T):
            start = max(0, t - window + 1)
            x = values[:, start:t + 1]  # (n, L)
            L = x.shape[1]
            w = decay ** np.arange(L - 1, -1, -1, dtype=np.float64)  # newest gets weight 1
            w /= w.sum()
            out[:, t] = x @ w        
        return out


    @staticmethod
    def moving_average(
        values: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        True finite-window rolling SMA (like the truncated EMA pattern).
        Uses only the last `window` points.
        Warmup uses shorter windows.
        values: (n_samples, n_timesteps)
        """
        values = np.asarray(values, dtype=np.float64)
        n, T = values.shape
        out = np.empty((n, T), dtype=np.float64)
        for t in range(T):
            start = max(0, t - window + 1)
            out[:, t] = values[:, start:t + 1].mean(axis=1)
        return out
    
    
    @staticmethod
    def colinearity_treenames_mask(
        X: np.ndarray,
        y: np.ndarray,
        treenames: np.ndarray,
        m_features: int,
        thr_crosscorr: float = 0.2,
    ) -> np.ndarray:
        """
        Select up to m_features that are:
        - highly correlated with y
        - not too correlated (>|thr_crosscorr|) with already selected features.
        """
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of rows")
        # Feature–feature correlation
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        corr[np.abs(corr) > 0.5] = 0.0
        # Feature–target correlations via sklearn
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_y = r_regression(X, y)        # shape (p,)
        corr_y = np.nan_to_num(corr_y, nan=0.0)
        corr_y[np.abs(corr_y) > 0.5] = 0.0
        # Order by descending |corr(feature, y)|
        order = np.argsort(-np.abs(corr_y))
        keep = np.zeros(p, dtype=bool)
        # Greedy selection
        for j in order:
            if keep.sum() >= m_features:
                break
            if not keep.any() or np.all(np.abs(corr[j, keep]) <= thr_crosscorr):
                keep[j] = True
                logger.debug(f"Selected feature {treenames[j]} with |corr|={np.abs(corr_y[j])}")
        return keep
    
    
    @staticmethod
    def maximize_through_quantile_windows(
        matrix: np.ndarray,
        y: np.ndarray,
        q_len: float,
        dates_digitized: np.ndarray,          # (n_samples,)
        weight_mean: float = 0.8,             # higher => win_means more important
        feat_name: str = "Feature",
    ):
        assert 0.0 < q_len <= 1.0, "q_len must be in (0,1]"
        assert 0.0 <= weight_mean <= 1.0, "weight_mean must be in [0,1]"
        
        n_samples, n_wndws = matrix.shape
        qwndw_ilen = np.clip(int((q_len + 1e-8) * (n_samples-1) + 1), 1, n_samples, dtype=int)
        qwndw = qwndw_ilen - 1
        
        if qwndw_ilen == 1:
            logger.error(f"  _maximize_through_quantile_windows: q_len too small, using q_len={1.0/(n_samples-1):.6f} instead.")
            raise ValueError("q_len too small for the number of samples.")
        
        # --- Sort each column once (vectorized) ---
        order = np.argsort(matrix, axis=0)  # (n_samples, n_wndws)
        y_sorted = np.take_along_axis(y[:, None], order, axis=0)  # (n_samples, n_wndws)
        d_sorted = np.take_along_axis(dates_digitized[:, None], order, axis=0)  # (n_samples, n_wndws)
        
        # --- Rolling mean of log(y) via cumulative sums (vectorized) ---
        logy = np.log(y_sorted) 
        cs = np.cumsum(logy, axis=0)
        cs = np.vstack([np.zeros((1, n_wndws)), cs]) 
        win_sums = cs[qwndw:] - cs[:-qwndw]           # (n_samples - qwndw, n_wndws)
        win_means = win_sums / qwndw
        
        # --- Rolling "date spread" score (std or variance) via cumulative sums ---
        d = d_sorted.astype(np.float64, copy=False)
        csd  = np.cumsum(d, axis=0)
        csd2 = np.cumsum(d * d, axis=0)
        csd  = np.vstack([np.zeros((1, n_wndws)), csd])
        csd2 = np.vstack([np.zeros((1, n_wndws)), csd2])
        
        d_sum  = csd[qwndw:]  - csd[:-qwndw]
        d2_sum = csd2[qwndw:] - csd2[:-qwndw]
        d_mean = d_sum / qwndw
        d_var  = d2_sum / qwndw - d_mean * d_mean
        d_var  = np.maximum(d_var, 0.0)
        date_score = np.sqrt(d_var)
        
        # --- Make scores comparable (per column) then combine ---
        def _zscore(a):
            mu = np.nanmean(a, axis=0, keepdims=True)
            sd = np.nanstd(a, axis=0, keepdims=True)
            return (a - mu) / (sd + 1e-12)

        wm_z = _zscore(win_means)
        ds_z = _zscore(date_score)

        w = float(weight_mean)
        score_weighted = w * wm_z + (1.0 - w) * ds_z

        arg = np.nanargmax(score_weighted, axis=0)            # (n_wndws,)
        idx_end = arg + qwndw
        
        # --- qlow/qhigh (vectorized) ---
        qlow  = np.clip((arg     - 0.5) / (n_samples-1), 0.0, 1.0)
        qhigh = np.clip((idx_end + 0.5) / (n_samples-1), 0.0, 1.0)
        
        # --- Quantile values using already-sorted features ---
        feat_sorted = np.take_along_axis(matrix, order, axis=0)  # (n_samples, n_wndws)
        
        def quantile_from_sorted(sorted_x, q):
            # sorted_x: (n, k), q: (k,)
            n = sorted_x.shape[0]
            k = sorted_x.shape[1]
            pos = q * (n - 1)
            lo = np.floor(pos).astype(int)
            hi = np.ceil(pos).astype(int)
            w = pos - lo # between 0 and 1. used for linear interp. shape (k,)
            cols = np.arange(k)
            x_lo = sorted_x[lo, cols]
            x_hi = sorted_x[hi, cols]
            return x_lo * (1.0 - w) + x_hi * w

        qlow_val  = quantile_from_sorted(feat_sorted, qlow)
        qhigh_val = quantile_from_sorted(feat_sorted, qhigh)
        
        # --- Mean y inside band (vectorized) ---
        mask_matrix: np.ndarray = (matrix >= qlow_val) & (matrix <= qhigh_val)  # (n_samples, n_wndws)
        denom = mask_matrix.sum(axis=0)
        num = (mask_matrix * np.log(y)[:, None]).sum(axis=0)
        ymean_vals = np.exp(np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float), where=(denom > 0)))
        
        # --- Logging feat---
        best_idx = int(np.nanargmax(ymean_vals))
        for i in range(len(ymean_vals)):
            logger.debug(
                f"  {feat_name} col {i} -> qlow: {qlow[i]:.4f} ({qlow_val[i]:.6f}) | "
                f"qhigh: {qhigh[i]:.4f} ({qhigh_val[i]:.6f}) | mean y: {ymean_vals[i]:.6f}"
            )
        logger.debug(
            f"  {feat_name} best idx {best_idx} -> qlow: {qlow[best_idx]:.4f} ({qlow_val[best_idx]:.6f}) | "
            f"qhigh: {qhigh[best_idx]:.4f} ({qhigh_val[best_idx]:.6f}) | mean y: {ymean_vals[best_idx]:.6f}"
        )
        
        return ymean_vals, qlow_val, qhigh_val, qlow, qhigh
