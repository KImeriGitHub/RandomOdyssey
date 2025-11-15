from typing import Iterable, Union
import numpy as np
import polars as pl
import lightgbm as lgb

from src.hyperparameterTuning.HelperMetrics import HelperMetrics

class HelperFunctions:
    def __init__():
        pass

    @staticmethod
    def optimize_sl_tp(
        arr: np.ndarray, 
        arr_low: np.ndarray,
        arr_high: np.ndarray,
        arr_open: np.ndarray,
        *,
        n_grid: int = 15,            # number of quantile points
        sl_max: float = 0.995,       # SL upper cap
        tp_min: float = 1.005,       # TP lower cap
        spread_cost: float = 0.001,
        commission: float = 0.0000,
    ) -> tuple[float, float, dict]:
        """
        Returns (best_sl, best_tp, info) maximizing mean(out) where:
            out = collapse_sl_tp(..., sl, tp, ...)

        SL candidates: [-inf, quantiles(arr[:,-1]) clipped to <= sl_max, sl_max]
        TP candidates: [tp_min, quantiles(arr[:,-1]) clipped to >= tp_min, +inf)

        Notes:
        - -inf for SL and +inf for TP are treated as "no stop" (passed as None).
        - Pairs violating common-sense constraints are skipped:
            if finite(sl) and sl >= 1.0  -> skip
            if finite(tp) and tp <= 1.0  -> skip
            if both finite and sl >= tp  -> skip
        """
        # --- candidate grids from quantiles of the last column ---
        last = np.concatenate((arr_low[:, -1], arr_high[:, -1]))
        squish_power = 1.3
        squish_pwr_inv = 1.0 / squish_power
        qs_sl = np.linspace(0.001**squish_pwr_inv, 0.495**squish_pwr_inv, n_grid)**squish_power
        qs_tp = 1.0 - qs_sl[::-1]

        qvals_sl = np.quantile(last, qs_sl)
        # SL: (-inf .. sl_max]
        stretch_factor = 0.99
        sl_cands = np.unique(
            np.concatenate((
                qvals_sl[qvals_sl <= sl_max]*stretch_factor,
                np.array([sl_max])
            ))
        )

        # TP: [tp_min .. +inf)
        qvals_tp = np.quantile(last, qs_tp)
        stretch_factor = 1.01
        tp_cands = np.unique(
            np.concatenate((
                np.array([tp_min]),
                qvals_tp[qvals_tp >= tp_min]*stretch_factor,
            ))
        )

        def _collapse(sl_val, tp_val):
            # interpret infinities as "no stop"
            sl_arg = np.min(last) if not np.isfinite(sl_val) else float(sl_val)
            tp_arg = np.max(last) if not np.isfinite(tp_val) else float(tp_val)
            return HelperMetrics.collapse_sl_tp(
                arr, arr_low, arr_high, arr_open,
                sl=sl_arg, tp=tp_arg, spread_cost=spread_cost, commission=commission
            )

        best = (-np.inf, None, None, None)  # (mean, sl, tp, std)

        for sl_val in sl_cands:
            for tp_val in tp_cands:
                out = _collapse(sl_val, tp_val)
                m = float(np.exp(np.mean(np.log(out))))
                if m > best[0]:
                    best = (m, float(sl_val), float(tp_val), float(np.std(out)))

        if best[1] is None:
            # fallback: no valid pair found -> use "no SL/TP"
            out = _collapse(-np.inf, np.inf)
            return -np.inf, np.inf, {"mean": float(np.mean(out)), "std": float(np.std(out)), "stage": "fallback"}

        mean_star, sl_star, tp_star, std_star = best
        return sl_star, tp_star, {
            "mean": mean_star,
            "std": std_star,
            "params": {
                "n_grid": n_grid, "sl_max": sl_max, "tp_min": tp_min,
                "spread_cost": spread_cost, "commission": commission
            }
        }
    
    @staticmethod
    def perfect_sl_tp(
        arr: np.ndarray,
        tp_min: float = 1.005,
        tp_buffer_pct: float = 0.1,
        sl_max: float = 0.995,
        sl_buffer_pct: float = 0.1,
        sl_min: float = 0.92,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized 'perfect' stop-loss and take-profit levels for row-wise time series.

        Parameters
        ----------
        arr : np.ndarray, shape (N, T)
            Row-wise time series of (normalized) prices.
        tp_min : float
            Minimum TP level to consider.
        tp_buffer_pct : float
            Buffer applied to TP: buffer = (tp - 1) * pct.
        sl_max : float
            Upper cap for SL (prevents SL > ~1).
        sl_buffer_pct : float
            Buffer applied to SL: buffer = (1 - price) * pct.
        sl_min : float
            Lower cap for SL (prevents unrealistically deep stops).

        Returns
        -------
        sl : np.ndarray, shape (N,)
            Stop-loss levels.
        tp : np.ndarray, shape (N,)
            Take-profit levels.
        """

        # --- Helpers
        def btp(val, pct):  # Buffer for take-profit
            return (val - 1.0) * pct

        def bsl(val, pct):  # Buffer for stop-loss
            return (1.0 - val) * pct

        # --- Take-profit
        arr_max = np.max(arr, axis=1)
        tp = np.maximum(arr_max, tp_min)

        tp_min_buffer_adj = tp_min + btp(tp_min, tp_buffer_pct)
        crossed = arr_max >= tp_min_buffer_adj

        # If we crossed the threshold, pull TP down by its buffer
        tp = np.where(crossed, tp - btp(tp, tp_buffer_pct), tp)

        # --- Stop-loss
        arr_argmax = np.argmax(arr, axis=1)
        cummin = np.minimum.accumulate(arr, axis=1)  # (N, T)
        mins_to_argmax = cummin[np.arange(arr.shape[0]), arr_argmax]  # (N,)
        last = arr[:, -1]

        # If TP-threshold crossed: set SL below the min up to argmax
        sl_cross = mins_to_argmax - bsl(mins_to_argmax, sl_buffer_pct)

        # If not crossed: set SL below the last price  (FIX: use minus, not plus)
        sl_else = last - bsl(last, sl_buffer_pct)

        sl = np.where(crossed, sl_cross, sl_else)

        # --- Safety clamps
        sl = np.minimum(sl, sl_max)           # never above sl_max
        if sl_min is not None:
            sl = np.maximum(sl, sl_min)       # never below floor

        # Optional: ensure SL < TP (small epsilon)
        eps = 1e-9
        sl = np.minimum(sl, tp - eps)

        return sl, tp

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