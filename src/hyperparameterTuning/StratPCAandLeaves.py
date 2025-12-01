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
from src.hyperparameterTuning.HelperFunctions import HelperFunctions
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

    def run(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        ytr_tree_low,
        ytr_tree_high,
        ytr_tree_open,
        Xte_tree,
        Xte_time,
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
                y_train=ytr_tree[:, -1],
                X_test=None,
                y_test=None,
            )
        except Exception as e:
            logger.disabled = False
            logger.warning(f"  LGB failed: {e}")
            return 1.0
        finally:
            logger.disabled = False

        tree_n_max = min(model_lgb.num_trees(), tree_n_max)
        labels_top, scores_top = HelperFunctions.top_leaf_labels_per_tree(
            model_lgb, Xtr_pca, ytr_tree[:, -1], tree_n_max=tree_n_max, top_n_max=max(1, top_n_max)
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

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(ytr_tree, ytr_tree_low, ytr_tree_high, ytr_tree_open)
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)
        
        score_tr = np.random.rand(Xtr_tree.shape[0])
        score_te = np.random.rand(Xte_tree.shape[0])

        return mask_train, mask_sel, sl_tr, sl_te, tp_tr, tp_te, score_tr, score_te

    def precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        ytr_tree_low,
        ytr_tree_high,
        ytr_tree_open,
        Xte_tree,
        Xte_time,
        treenames,
        timenames,
        meta_train,
        meta_test,
    ) -> tuple[np.ndarray, np.ndarray]:
        if treenames is None or meta_train is None or meta_test is None:
            raise ValueError("treenames, meta_train and meta_test are required.")

        params = dict(self.precompute_params)

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree[:, -1],
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _geometric_mean_safe(self, arr):
        arr = np.asarray(arr, dtype=float)
        minv = np.min(arr) if arr.size else 0.0
        shift = -minv + 1e-9 if minv <= 0 else 0.0
        return float(np.exp(np.mean(np.log(arr + shift)))) if arr.size else np.nan


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