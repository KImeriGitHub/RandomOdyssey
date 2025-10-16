import numpy as np
import optuna
import polars as pl
import datetime
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta

import logging
logger = logging.getLogger(__name__)

class StratLGBLeavesTree(BaseStrategy):
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
        "FilterSamples_cat_highestShareholderEquity_q0.6": False,
        "FilterSamples_cat_volatility_qdown0.025": False,
        "FilterSamples_cat_predictability_qup0.9": False,
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
        opt_params["LGB_lambda_l1"]                 = trial.suggest_float("LGB_lambda_l1", 0.001, 2.9, log=True)
        opt_params["LGB_lambda_l2"]                 = trial.suggest_float("LGB_lambda_l2", 0.001, 1.0, log=True)
        opt_params["LGB_feature_fraction"]          = trial.suggest_float("LGB_feature_fraction", 0.2, 0.79, log=True)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 200, 2050, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 3, 30, step=1)
        opt_params["LGB_learning_rate"]             = 0.1 #trial.suggest_float("LGB_learning_rate", 1e-4, 2e-0, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 20, 950, step=10)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 1e-6, 1e-0, log=True)
        opt_params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 1e-2, 5e-1, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = trial.suggest_float("LGB_min_sum_hessian_in_leaf", 5e-4, 1e-0, log=True)
        opt_params["LGB_max_bin"]                   = trial.suggest_int("LGB_max_bin", 25, 605, step=10)
        opt_params["LGB_early_stopping_rounds"]     = 20

        #opt_params["n_training_days"]   = trial.suggest_int("n_training_days", 400, 900, step=100)
        opt_params["do_transform"]      = trial.suggest_categorical("do_transform", [True, False])
        opt_params["tree_n_max"]        = 1 #trial.suggest_int("tree_n_max", 5, 75, step=5)
        opt_params["min_n_tar"]         = 100 #trial.suggest_int("min_n_tar", 1, 5)
        opt_params["top_n_max"]         = 200 #trial.suggest_int("top_n_max", 3, 25)  

        params = dict(self.base_params)
        params.update(opt_params)
        return params

    def run(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        treenames,
        timenames,
        meta_train,
        meta_test,
        opt_params: dict,
    ) -> float:
        do_transform =      opt_params["do_transform"]
        tree_n_max =        opt_params["tree_n_max"]
        min_n_tar =         opt_params["min_n_tar"]
        top_n_max =         opt_params["top_n_max"]
        mm: MachineModels = MachineModels(opt_params)

        Xd_tr, ytr_tree = Xtr_tree, ytr_tree
        Xd_te, yte_tree = Xte_tree, yte_tree

        sl_val, tp_val = HelperFunctions.optimal_sl_tp(ytr_tree)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        logger.debug(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.debug(f"   ytr_tree: n={ytr_tree.size}, mean={ytr_tree.mean():.6f}, std={ytr_tree.std():.6f}")

        if do_transform:
            scaler = StandardScaler().fit(Xd_tr)
            Xd_tr = scaler.transform(Xd_tr)
            Xd_te = scaler.transform(Xd_te)

        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xd_tr,
                y_train=ytr_tree[:, -1],
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
        labels_top, scores_top = HelperFunctions.top_leaf_labels_per_tree(
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

        return mask_sel, sl_te, tp_te

    def precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
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

        sl_val, tp_val = HelperFunctions.optimal_sl_tp(ytr_tree)
        sl_tr_vec = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te_vec = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr_vec = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te_vec = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        logger.info(
            "  Precompute -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(
            "  Precompute -> sl %.2f%% | tp: %.2f%%",
            sl_val,
            tp_val,
        )

        return mask_train, mask_test, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _geometric_mean_safe(self, arr):
        arr = np.asarray(arr, dtype=float)
        minv = np.min(arr) if arr.size else 0.0
        shift = -minv + 1e-9 if minv <= 0 else 0.0
        return float(np.exp(np.mean(np.log(arr + shift)))) if arr.size else np.nan

    @staticmethod
    def _parse_params(trial: optuna.Trial, space: dict) -> dict:
        out = {}
        for name, spec in space.items():
            kind, lo, hi, kw = spec
            suggest = trial.suggest_int if kind == "int" else trial.suggest_float
            out[name] = suggest(name.replace("FilterSamples_", ""), lo, hi, **kw)
        return out
