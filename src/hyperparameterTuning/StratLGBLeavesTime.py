import numpy as np
import optuna
import polars as pl
import datetime
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.hyperparameterTuning.HelperMetrics import HelperMetrics
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

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
        "FilterSamples_cat_over10": True,
        "FilterSamples_cat_under5000": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.2": True,
        "FilterSamples_cat_volatility_qdown0.02": False,
        "FilterSamples_cat_volatility_qup0.975": False,
        "FilterSamples_cat_predictability_qup0.9": False,
        "FilterSamples_cat_predictability_qdown0.1": False,
    }

    base_params = {}

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {}
        opt_params["LGB_num_boost_round"]           = 1 #trial.suggest_int("LGB_num_boost_round", 40, 60, step=1)
        opt_params["LGB_lambda_l1"]                 = trial.suggest_float("LGB_lambda_l1", 0.001, 0.9, log=True)
        opt_params["LGB_lambda_l2"]                 = trial.suggest_float("LGB_lambda_l2", 0.0001, 0.1, log=True)
        opt_params["LGB_feature_fraction"]          = 1.0 #trial.suggest_float("LGB_feature_fraction", 0.90, 1.0, log=True)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 200, 1500, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 30, 120, step=1)
        opt_params["LGB_learning_rate"]             = 0.1 #trial.suggest_float("LGB_learning_rate", 1e-3, 1e-1, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 10, 100, step=5)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 0.0000005, 0.00005, log=True)
        opt_params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 1e-2, 5e-1, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.1, 1.5, log=True)
        opt_params["LGB_max_bin"]                   = trial.suggest_int("LGB_max_bin", 200, 1500, step=25)
        opt_params["LGB_early_stopping_rounds"]     = 20

        opt_params["t_win"]             = trial.suggest_int("t_win", 20, 50, step=1)
        #opt_params["n_training_days"]   = trial.suggest_int("n_training_days", 400, 900, step=100)
        opt_params["do_transform"]      = True #trial.suggest_categorical("do_transform", [True, False])
        opt_params["tree_n_max"]        = 1 #trial.suggest_int("tree_n_max", 5, 75, step=5)
        opt_params["min_n_tar"]         = 10 #trial.suggest_int("min_n_tar", 2, 5)
        opt_params["top_n_max"]         = trial.suggest_int("top_n_max", 200, 500)  
        
        opt_params["cat_ratio"] = trial.suggest_float("cat_ratio", 0.20, 0.35)
        opt_params["vol_down_q"] = False #trial.suggest_categorical("vol_down_q", [True, False])

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
        t_win =             opt_params["t_win"]
        do_transform =      opt_params["do_transform"]
        tree_n_max =        opt_params["tree_n_max"]
        min_n_tar =         opt_params["min_n_tar"]
        top_n_max =         opt_params["top_n_max"]
        
        cat_ratio =         opt_params["cat_ratio"]
        opt_params[f"FilterSamples_cat_volatility_qdown{cat_ratio:.2f}"] = opt_params["vol_down_q"]
        opt_params[f"FilterSamples_cat_volatility_qup{(1-cat_ratio):.2f}"] = not opt_params["vol_down_q"]
        mm: MachineModels = MachineModels(opt_params)
        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        fs_pre = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=np.max(ytr_tree, axis=1),
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=None,
            meta_train=meta_train,
            meta_test=meta_test,
            params=opt_params,
        )

        cat_train, cat_test = fs_pre.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test

        Xd_tr, bad_tr = self._make_design(Xtr_time[mask_train], t_win)
        Xd_te, bad_te = self._make_design(Xte_time[mask_test], t_win)
        keep_tr = ~bad_tr
        keep_te = ~bad_te
        Xd_tr, ytr_tree, ytr_tree_low, ytr_tree_high, ytr_tree_open = Xd_tr[keep_tr], ytr_tree[mask_train][keep_tr], ytr_tree_low[mask_train][keep_tr], ytr_tree_high[mask_train][keep_tr], ytr_tree_open[mask_train][keep_tr]
        Xd_te = Xd_te[keep_te]
        
        ytr_tree_opt = np.max(ytr_tree, axis=1)

        logger.debug(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.info(f"   ytr_tree: n={ytr_tree_opt.size}, mean={ytr_tree_opt.mean():.6f}, std={ytr_tree_opt.std():.6f}")

        if do_transform:
            ss = StandardScaler()
            Xd_tr = ss.fit_transform(Xd_tr)
            Xd_te = ss.transform(Xd_te)

        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xd_tr,
                y_train=ytr_tree_opt,
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
            model_lgb, Xd_tr, ytr_tree_opt, tree_n_max=tree_n_max, top_n_max=max(1, top_n_max or 1)
        )
        n_trees = labels_top.shape[1]
        
        # Top labels by score
        leaf_tr = model_lgb.predict(Xd_tr, pred_leaf=True)
        leaf_tr = leaf_tr.reshape(-1, 1) if leaf_tr.ndim == 1 else leaf_tr  # shape (n_samples, n_trees)
        leaf_tr = leaf_tr[:, :n_trees]  # restrict to used trees
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
        mask_tr = np.zeros(Xd_tr.shape[0], dtype=bool)
        for i in range(len(r_idx)):
            r = r_idx[i]
            t = t_idx[i]
            lbl = labels_top[r, t]
            if lbl == -1:
                continue
            mask_sel |= (leaf_te[:, t] == lbl)
            mask_tr |= (leaf_tr[:, t] == lbl)
            sel_pairs.append((int(lbl), int(t), int(r)))
            if mask_sel.sum() >= min_n_tar:
                break
        
        res_mask = np.zeros(Xte_tree.shape[0], dtype=bool)
        keep_tmp = mask_test.copy()
        keep_tmp[mask_test] = keep_te
        res_mask[keep_tmp] = mask_sel

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_tr], 
            ytr_tree_low[mask_tr], 
            ytr_tree_high[mask_tr], 
            ytr_tree_open[mask_tr],
            n_grid = 7
        )
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)
        
        score_tr = np.random.rand(Xtr_tree.shape[0])
        score_te = np.random.rand(Xte_tree.shape[0])

        return mask_train, res_mask, sl_tr, sl_te, tp_tr, tp_te, score_tr, score_te

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

        params = self.precompute_params

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        fs_pre = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree[:, -1],
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=None,
            meta_train=meta_train,
            meta_test=meta_test,
            params=params,
        )

        cat_train, cat_test = fs_pre.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid = 6
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    def _make_design(self, X, t_win):
        Xw: np.ndarray = X[:, -(t_win+1):, 0:5].copy()
        Xw = (Xw-0.5)*2.0
        
        mask_bad = np.zeros(Xw.shape[0], dtype=bool)
        bound_bad = 1 - np.tanh(1 - 1e-4)
        mask_bad = np.any((Xw[:,:,0:4] <= (-1+bound_bad)) | (Xw[:,:,0:4] >= (1-bound_bad)), axis=(1,2))
        Xw[:,:,0:4] = np.clip(Xw[:,:,0:4], -1+bound_bad, 1-bound_bad)
        
        Xw[:,:,0:4] = np.arctanh(Xw[:,:,0:4]) + 1.0
        
        return Xw.reshape(Xw.shape[0], -1), mask_bad
