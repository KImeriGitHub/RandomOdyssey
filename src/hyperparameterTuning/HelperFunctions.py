from typing import Iterable, Union
import numpy as np
import polars as pl
import lightgbm as lgb

class HelperFunctions:
    def __init__():
        pass

    @staticmethod
    def optimal_sl_tp(arr: np.ndarray, rounding_digits: int = 3) -> tuple[float, float]:
        """
        PRE:
            arr: n x d array, values > 0, around 1.0.
                 arr[i,-1] is latest; arr[i,0] is earliest
            rounding_digits: accuracy of sl and tp, for speed up
        POST:
            (sl_val, tp_val): stop-loss (<≈1) and take-profit (>≈1) that
            maximize the geometric mean of the realized outcome defined by:
              - if any arr[i, j] < sl -> out[i] = sl
              - elif any arr[i, j] > tp -> out[i] = tp
              - else out[i] = arr[i, -1]
        Notes:
            We restrict the search to sl ∈ (0, 1] and tp ∈ [1, max(arr)],
            and to candidate values at row-wise extrema where regime changes occur.
        """
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError("arr must be a non-empty 2D numpy array")

        # Row statistics that fully determine regime for any (sl, tp)
        row_min = arr.min(axis=1)
        row_max = arr.max(axis=1)
        row_last = arr[:, -1]

        # Domains (conservative & practical):
        overall_max = float(row_max.max())
        # Candidate SL values: unique row minima clipped to <=1, plus 1.0
        sl_candidates = np.unique(np.clip(
            np.round(row_min, rounding_digits), a_min=None, a_max=1.0
        ))
        if 1.0 not in sl_candidates:
            sl_candidates = np.sort(np.append(sl_candidates, 1.0))

        # Candidate TP values: unique row maxima clipped to >=1, plus 1.0
        tp_candidates = np.unique(np.clip(
            np.round(row_max, rounding_digits), a_min=1.0, a_max=None
        ))
        if 1.0 not in tp_candidates:
            tp_candidates = np.sort(np.append(tp_candidates, 1.0))

        # (Optional) include the "no-TP-trigger" ceiling (overall_max)
        if overall_max not in tp_candidates:
            tp_candidates = np.sort(np.append(tp_candidates, overall_max))

        best_sl, best_tp = 1.0, overall_max
        best_obj = -np.inf

        # Vectorized evaluation over candidate grid
        for sl in sl_candidates:
            # Precompute SL mask once per sl
            sl_mask = row_min < sl
            for tp in tp_candidates:
                # Enforce sensible band: sl <= 1 <= tp
                if sl > 1.0 or tp < 1.0:
                    continue

                tp_mask = (~sl_mask) & (row_max > tp)
                # Outcomes per rule
                out = np.where(sl_mask, sl, np.where(tp_mask, tp, row_last))

                # Geometric mean ↔ maximize sum(log(out))
                # (arr > 0 ensures out > 0)
                obj = float(np.mean(np.log(out)))
                if obj > best_obj:
                    best_obj = obj
                    best_sl, best_tp = float(sl), float(tp)

        return best_sl, best_tp

    @staticmethod
    def top_leaf_labels_per_tree(self,
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