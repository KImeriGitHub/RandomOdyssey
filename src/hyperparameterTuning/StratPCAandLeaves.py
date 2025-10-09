import numpy as np
import optuna
import polars as pl
import datetime
import lightgbm as lgb

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta

import logging
logger = logging.getLogger(__name__)

class StratPCAandLeaves(BaseStrategy):
    expected_load_params = {
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }
    precompute_params = {
        "FilterSamples_cat_over20.0": True,
        "FilterSamples_cat_under2000.0": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_doubleFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.9": True,
        
        "FilterSamples_q_up": 0.9914,
        "FilterSamples_method": "taylor",

        "FilterSamples_days_to_train_end": 10,
        "FilterSamples_lincomb_epochs": 8,
        "FilterSamples_lincomb_lr": 0.000009,
        "FilterSamples_lincomb_probs_noise_std": 0.057207,
        "FilterSamples_lincomb_show_progress": False,
        "FilterSamples_lincomb_subsample_ratio": 0.527366,
        "FilterSamples_lincomb_sharpness": 0.78169,
        "FilterSamples_lincomb_featureratio": 0.8,
        "FilterSamples_lincomb_itermax": 1,
        "FilterSamples_lincomb_init_toprand":  3,
        "FilterSamples_lincomb_batch_size": 2**12,
        
        "FilterSamples_taylor_horizon_days": 60,
        "FilterSamples_taylor_roll_window_days": 15,
        "FilterSamples_taylor_weight_slope": 0.43,
    }

    base_params = {}

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {}
        opt_params["LGB_num_boost_round"]           = 100 #trial.suggest_int("LGB_num_boost_round", 40, 60, step=1)
        opt_params["LGB_lambda_l1"]                 = 5e-1 #trial.suggest_float("LGB_lambda_l1", 5e-3, 1e-1, log=True)
        opt_params["LGB_lambda_l2"]                 = 5e-1 #trial.suggest_float("LGB_lambda_l2", 1e-5, 1e-3, log=True)
        opt_params["LGB_feature_fraction"]          = trial.suggest_float("LGB_feature_fraction", 0.5, 0.7)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 50, 100, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 3, 6, step=1)
        opt_params["LGB_learning_rate"]             = trial.suggest_float("LGB_learning_rate", 1e-4, 0.003, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 800, 1050, step=10)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 1e-5, 0.5, log=True)
        opt_params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 1e-2, 5e-1, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = trial.suggest_float("LGB_min_sum_hessian_in_leaf", 5e-3, 1e-0, log=True)
        opt_params["LGB_max_bin"]                   = trial.suggest_int("LGB_max_bin", 25, 200, step=5)
        opt_params["LGB_early_stopping_rounds"]     = 10

        opt_params["n_components"]       = trial.suggest_int("n_components", 5, 20, step = 5)
        opt_params["tree_n_max"]        = trial.suggest_int("tree_n_max", 25, 75, step=5)
        opt_params["min_n_tar"]         = trial.suggest_int("min_n_tar", 0, 5)
        opt_params["top_n_max"]         = 5 #trial.suggest_int("top_n_max", 3, 13)  

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
        n_components  = opt_params["n_components"]
        tree_n_max    = opt_params["tree_n_max"]
        min_n_tar     = opt_params["min_n_tar"]
        top_n_max     = opt_params["top_n_max"]

        mm: MachineModels = MachineModels(opt_params)

        Xtr_clean, Xte_clean, keep_mask = _clean_train_test(Xtr_tree, Xte_tree)
        Xtr_pca, Xte_pca = _pca(Xtr_clean, Xte_clean, n_comp=n_components, svd_solver="arpack")

        logger.info(f"  PCA reduced tree features from (TRAIN) {Xtr_tree.shape[1]} to {Xtr_pca.shape[1]} dimensions.")
        logger.info(f"  PCA reduced tree features from (TEST) {Xte_tree.shape[1]} to {Xte_pca.shape[1]} dimensions.")

        n_comp = Xtr_pca.shape[1]

        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xtr_pca,
                y_train=ytr_tree,
                X_test=Xte_pca,
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
            model_lgb, Xtr_pca, ytr_tree, tree_n_max=tree_n_max, top_n_max=max(1, top_n_max)
        )
        n_trees = labels_top.shape[1]
        
        # Top labels by score
        leaf_te = model_lgb.predict(Xte_pca, pred_leaf=True)
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

        params = dict(self.precompute_params)

        cat_mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        cat_mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

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
        cat_mask_train &= cat_train
        if cat_test is not None:
            cat_mask_test &= cat_test

        ### MAIN FILTERING
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

        #if params["FilterSamples_method"] == "taylor":
        #    mask_train, mask_test = fs.taylor_feature_masks()
        #if params["FilterSamples_method"] == "lincomb":
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

    def _top_leaf_labels_per_tree(self,
        model: lgb.Booster,
        X,
        y,
        tree_n_max: int,
        top_n_max: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For each tree t in [0, tree_idx_max], compute your score per leaf.
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
                pl.std("y").alias("std_y"),
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

def _clean_train_test(
    Xtr_tree: np.ndarray,
    Xte_tree: np.ndarray,
    var_thresh: float = 1e-12,
    winsor_z: float | None = 6.0,
    scale: bool = True,
):
    """
    Pre-clean Xtr_tree/Xte_tree consistently:
      1) replace non-finite with NaN
      2) median-impute (fit on train)
      3) drop near-zero variance cols (by train variance)
      4) optional winsorize by train z-score (clip to ±winsor_z)
      5) optional standardize (fit on train)

    Returns:
      Xtr_clean, Xte_clean, keep_mask  (keep_mask maps kept feature columns)
    """
    Xtr = np.asarray(Xtr_tree, dtype=np.float64)
    Xte = np.asarray(Xte_tree, dtype=np.float64)

    # 1) finite
    Xtr = Xtr.copy(); Xte = Xte.copy()
    Xtr[~np.isfinite(Xtr)] = np.nan
    Xte[~np.isfinite(Xte)] = np.nan

    # 2) impute (fit on train)
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr)
    Xte = imp.transform(Xte)

    # 3) variance threshold (on train)
    var = Xtr.var(axis=0)
    keep_mask = var > var_thresh
    if not np.any(keep_mask):
        # fallback: keep at least one column
        keep_mask = var == var.max()
    Xtr = Xtr[:, keep_mask]
    Xte = Xte[:, keep_mask]

    # 4) winsorize (clip by train z-scores)
    if winsor_z is not None:
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        def _winsor(X):
            Z = (X - mu) / sd
            Z = np.clip(Z, -winsor_z, winsor_z)
            return Z * sd + mu
        Xtr = _winsor(Xtr)
        Xte = _winsor(Xte)

    # 5) scale (fit on train)
    if scale:
        sc = StandardScaler(with_mean=True, with_std=True)
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    return Xtr, Xte, keep_mask

def _pca(Xtr, Xte, n_comp=5, svd_solver="full"):
    # 1) fit once to get variance curve
    p = PCA(n_components=n_comp, svd_solver=svd_solver)
    p.fit(Xtr)
    Xtr_pca = p.transform(Xtr)
    Xte_pca = p.transform(Xte)

    Xtr_pca = Xtr_pca.reshape(-1,1) if Xtr_pca.ndim == 1 else Xtr_pca
    Xte_pca = Xte_pca.reshape(-1,1) if Xte_pca.ndim == 1 else Xte_pca

    return Xtr_pca, Xte_pca