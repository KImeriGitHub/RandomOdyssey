import numpy as np
import optuna
import polars as pl
import datetime
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta

import logging
logger = logging.getLogger(__name__)

class StratLGBLeavesTime(BaseStrategy):
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

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {}
        opt_params["LGB_num_boost_round"]           = 5 #trial.suggest_int("LGB_num_boost_round", 40, 60, step=1)
        opt_params["LGB_lambda_l1"]                 = 0.001 #trial.suggest_float("LGB_lambda_l1", 1e-5, 2e-0, log=True)
        opt_params["LGB_lambda_l2"]                 = 0.001 #trial.suggest_float("LGB_lambda_l2", 1e-5, 2e-0, log=True)
        opt_params["LGB_feature_fraction"]          = trial.suggest_float("LGB_feature_fraction", 0.90, 1.0, log=True)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 1200, 3500, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 3, 31, step=2)
        opt_params["LGB_learning_rate"]             = 0.1 #trial.suggest_float("LGB_learning_rate", 1e-3, 1e-1, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 50, 2000, step=50)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 1e-5, 5e-1, log=True)
        opt_params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 1e-2, 5e-1, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = trial.suggest_float("LGB_min_sum_hessian_in_leaf", 1e-3, 5e-1, log=True)
        opt_params["LGB_max_bin"]                   = trial.suggest_int("LGB_max_bin", 50, 950, step=50)
        opt_params["LGB_early_stopping_rounds"]     = 20

        opt_params["t_win"]             = trial.suggest_int("t_win", 3, 22)
        #opt_params["n_training_days"]   = trial.suggest_int("n_training_days", 400, 900, step=100)
        opt_params["do_transform"]      = False #trial.suggest_categorical("do_transform", [True, False])
        opt_params["tree_n_max"]        = 1 #trial.suggest_int("tree_n_max", 5, 75, step=5)
        opt_params["min_n_tar"]         = 0
        opt_params["top_n_max"]         = trial.suggest_int("top_n_max", 5, 20)  

        params = dict(self.base_params)
        params.update(opt_params)
        return params

    def score(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        yte_tree,
        treenames,
        timenames,
        meta_train,
        meta_test,
        opt_params: dict,
    ) -> float:
        t_win =             opt_params["t_win"]
        do_transform =      opt_params["do_transform"]
        tree_n_max =        opt_params["tree_n_max"]
        min_n_tar =         opt_params["min_n_tar"]
        top_n_max =         opt_params["top_n_max"]
        mm: MachineModels = MachineModels(opt_params)

        Xd_tr, bad_tr = self._make_design(Xtr_time, t_win)
        Xd_te, bad_te = self._make_design(Xte_time, t_win)
        keep_tr = ~bad_tr
        keep_te = ~bad_te
        Xd_tr, ytr_tree = Xd_tr[keep_tr], ytr_tree[keep_tr]
        Xd_te, yte_tree = Xd_te[keep_te], yte_tree[keep_te]

        logger.debug(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.debug(f"   ytr_tree: n={ytr_tree.size}, mean={ytr_tree.mean():.6f}, std={ytr_tree.std():.6f}")

        if do_transform:
            ss = StandardScaler()
            Xd_tr = ss.fit_transform(Xd_tr)
            Xd_te = ss.transform(Xd_te)

        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xd_tr,
                y_train=ytr_tree,
                X_test=Xd_te,
                y_test=yte_tree,
            )
        except Exception as e:
            logger.disabled = False
            logger.warning(f"  LGB failed: {e}")
            return 1.0
        finally:
            logger.disabled = False

        tree_n_max = min(model_lgb.num_trees(), tree_n_max)
        labels_top, scores_top = self._top_leaf_labels_per_tree(
            model_lgb, Xd_tr, ytr_tree, tree_n_max=tree_n_max, top_n_max = max(1, top_n_max or 1)
        )
        n_trees = labels_top.shape[1]
        
        # Top labels by score
        leaf_te = model_lgb.predict(Xd_te, pred_leaf=True)
        leaf_te = leaf_te.reshape(-1, 1) if leaf_te.ndim == 1 else leaf_te  # shape (n_samples, n_trees)
        leaf_te = leaf_te[:, :n_trees]  # restrict to used trees

        # If everything is -1 across all ranks, bail out
        if labels_top.size == 0 or np.all(labels_top == -1):
            logger.warning("  LGB failed to generate predictions.")
            return 1.0
        
        # Rank every (rank, tree) pair by descending score
        r_idx, t_idx = np.unravel_index(np.argsort(scores_top.ravel())[::-1], scores_top.shape)
        
        sel_pairs = []  # (tree_idx, rank_idx)
        mask_sel = np.zeros(leaf_te.shape[0], dtype=bool)
        for i in range(len(r_idx)):
            r = r_idx[i]
            t = t_idx[i]
            lbl = labels_top[r, t]
            if lbl == -1:
                continue
            mask_sel |= (leaf_te[:, t] == lbl)
            sel_pairs.append((int(lbl), int(t), int(r)))
            if mask_sel.sum() >= min_n_tar:
                break
        y_selected = yte_tree[mask_sel]

        logger.info(
            f"  LGB selected {y_selected.size} out of {yte_tree.size} samples "
            f"using {len(sel_pairs)} (lbl, tree,rank) pairs: {sel_pairs}"
        )

        if y_selected.size == 0:
            logger.warning("  LGB failed to select testing values.")
            return 1.0

        score = self._geometric_mean_safe(y_selected)
        logger.info(
            f"score={score:.6f}, selected={y_selected.size}/{yte_tree.size}"
        )

        return float(score) if np.isfinite(score) else 1.0

    def mask_precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        yte_tree,
        treenames,
        timenames,
        meta_train,
        meta_test,
    ) -> tuple[np.ndarray, np.ndarray]:
        if treenames is None or meta_train is None or meta_test is None:
            raise ValueError("treenames, meta_train and meta_test are required.")

        params = self.precompute_params

        cat_mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        cat_mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        fs_pre = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree,
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=yte_tree,
            meta_train=meta_train,
            meta_test=meta_test,
            params=params,
        )

        cat_train, cat_test = fs_pre.categorical_masks()
        cat_mask_train &= cat_train
        if cat_test is not None:
            cat_mask_test &= cat_test

        #### MAIN FILTERING
        #fs = FilterSamples(
        #    Xtree_train = Xtr_tree[cat_mask_train], 
        #    ytree_train = ytr_tree[cat_mask_train], 
        #    treenames   = treenames,
        #    Xtree_test  = Xte_tree[cat_mask_test],
        #    ytree_test  = yte_tree[cat_mask_test],
        #    meta_train  = meta_train.filter(pl.Series(cat_mask_train)), 
        #    meta_test   = meta_test.filter(pl.Series(cat_mask_test)), 
        #    params      = params,
        #)

        #if self.base_params["FilterSamples_method"] == "taylor":
        #    mask_train, mask_test = fs.taylor_feature_masks()
        #if self.base_params["FilterSamples_method"] == "lincomb":
        #    mask_train, mask_test = fs.lincomb_masks()

        #score_train = fs.evaluate_mask(mask_train, 
        #    meta_train.filter(pl.Series(cat_mask_train))['date'], ytr_tree[cat_mask_train])
        #score_test  = fs.evaluate_mask(mask_test,  
        #    meta_test.filter(pl.Series(cat_mask_test))['date'], yte_tree[cat_mask_test])
        #logger.info(f"  Filtering Score (train) = {score_train}")
        #logger.info(f"  Filtering Score (test)  = {score_test}")

        #cat_mask_train[cat_mask_train] = mask_train
        #if mask_test is not None:
        #    cat_mask_test[cat_mask_test] = mask_test

        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * cat_mask_train.mean(),
            100 * cat_mask_test.mean(),
        )

        return cat_mask_train, cat_mask_test

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _geometric_mean_safe(self, arr):
        arr = np.asarray(arr, dtype=float)
        minv = np.min(arr) if arr.size else 0.0
        shift = -minv + 1e-9 if minv <= 0 else 0.0
        return float(np.exp(np.mean(np.log(arr + shift)))) if arr.size else np.nan
    
    def _make_design(self, X, t_win):
        Xw: np.ndarray = X[:, -(t_win+1):, 0:5].copy()
        Xw = (Xw-0.5)*2.0
        
        mask_bad = np.zeros(Xw.shape[0], dtype=bool)
        bound_bad = 1 - np.tanh(1 - 1e-4)
        mask_bad = np.any((Xw[:,:,0:4] <= (-1+bound_bad)) | (Xw[:,:,0:4] >= (1-bound_bad)), axis=(1,2))
        Xw[:,:,0:4] = np.clip(Xw[:,:,0:4], -1+bound_bad, 1-bound_bad)
        
        Xw[:,:,0:4] = np.arctanh(Xw[:,:,0:4]) + 1.0
        
        return Xw.reshape(Xw.shape[0], -1), mask_bad

    def _top_leaf_labels_per_tree(self,
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
                        .otherwise(1.0)  # so (1.0 - 1.0) → 0
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
