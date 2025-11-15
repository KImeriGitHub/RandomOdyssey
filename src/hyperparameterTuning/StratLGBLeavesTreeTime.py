import numpy as np
import optuna
import polars as pl
import datetime
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta

import logging
logger = logging.getLogger(__name__)

class StratLGBLeavesTreeTime(BaseStrategy):
    expected_load_params = {
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "FilterSamples_cat_over10": False,
        "FilterSamples_cat_under5000": False,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.2": False,
        "FilterSamples_cat_volatility_qdown0.02": False,
        "FilterSamples_cat_volatility_qup0.975": False,
        "FilterSamples_cat_predictability_qup0.9": False,
        "FilterSamples_cat_predictability_qdown0.1": False,
        
        "volatility_w": 56,
        "volatility_dir": "qup",
        "volatility_q": 0.533757,
        
        "volprice_q": 0.189539,
        "volprice_w": 19,
        
        "predictability_w": 52,
        "predictability_dir": "qdown",
        "predictability_q": 0.209829,
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
        opt_params["LGB_lambda_l1"]                 = trial.suggest_float("LGB_lambda_l1", 0.0001, 0.005, log=True)
        opt_params["LGB_lambda_l2"]                 = trial.suggest_float("LGB_lambda_l2", 0.00001, 0.0005, log=True)
        opt_params["apply_feature_fraction"]        = True #trial.suggest_categorical("apply_feature_fraction", [True, False])
        if opt_params["apply_feature_fraction"]:
            opt_params["LGB_feature_fraction"]      = trial.suggest_float("LGB_feature_fraction", 0.95, 0.97, log=False)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 745, 1900, step=5)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 17, 27, step=1)
        opt_params["LGB_learning_rate"]             = 0.1 #trial.suggest_float("LGB_learning_rate", 1e-4, 2e-0, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 5, 30, step=1)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 1e-7, 5e-5, log=True)
        opt_params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 1e-2, 5e-1, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.1, 0.5, log=True)
        opt_params["LGB_max_bin"]                   = trial.suggest_int("LGB_max_bin", 200, 350, step=10)
        opt_params["LGB_early_stopping_rounds"]     = 20

        #opt_params["n_training_days"]   = trial.suggest_int("n_training_days", 400, 900, step=100)
        opt_params["do_transform"]      = True #trial.suggest_categorical("do_transform", [True, False])
        opt_params["tree_n_max"]        = 1 #trial.suggest_int("tree_n_max", 5, 75, step=5)
        opt_params["min_n_tar"]         = 25 #trial.suggest_int("min_n_tar", 1, 5)
        opt_params["top_n_max"]         = trial.suggest_int("top_n_max", 350, 950, step=25)  
                
        opt_params["inc_FeatureTA"] = False #trial.suggest_categorical("inc_FeatureTA", [True, False])
        opt_params["inc_GroupDynamics"] = False #trial.suggest_categorical("inc_GroupDynamics", [True, False])
        opt_params["inc_Categorical"] = True #trial.suggest_categorical("inc_Categorical", [True, False])
        opt_params["inc_Financials"] = True #trial.suggest_categorical("inc_Financials", [True, False])
        opt_params["inc_Mathematical"] = True #trial.suggest_categorical("inc_Mathematical", [True, False])
        opt_params["inc_Seasonal"] = True #trial.suggest_categorical("inc_Seasonal", [True, False])
        opt_params["exc_lag"] = True
        
        opt_params["ytree_kind"] = "abslast" #trial.suggest_categorical("ytree_kind", ["last", "abslast"]) #mean and max not very good

        opt_params["t_win"] = trial.suggest_int("t_win", 15, 30, step=1)

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
        mm: MachineModels = MachineModels(opt_params)
            
        logger.info(f"  Before filtering: tr {Xtr_tree.shape}, te {Xte_tree.shape}")
            
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        if opt_params["ytree_kind"] == "last":
            ytr_tree_opt = ytr_tree[:, -1]
        elif opt_params["ytree_kind"] == "mean":
            ytr_tree_opt = np.mean(ytr_tree, axis=1)
        elif opt_params["ytree_kind"] == "abslast":
            ytr_tree_opt = np.abs(ytr_tree[:, -1])
        elif opt_params["ytree_kind"] == "max":
            ytr_tree_opt = np.max(ytr_tree, axis=1)

        tn = np.asarray(treenames, dtype=str)
        mask_treenames = np.zeros(len(treenames), dtype=bool)
        if opt_params.get("inc_FeatureTA"):
            mask_treenames |= np.char.find(tn, "FeatureTA_") >= 0
        if opt_params.get("inc_GroupDynamics"):
            mask_treenames |= np.char.find(tn, "FeatureGroup_") >= 0
        if opt_params.get("inc_Categorical"):
            mask_treenames |= np.char.find(tn, "Category_") >= 0
        if opt_params.get("inc_Financials"):
            mask_treenames |= np.char.find(tn, "FinData_") >= 0
        if opt_params.get("inc_Mathematical"):
            mask_treenames |= np.char.find(tn, "MathFeature_") >= 0
        if opt_params.get("inc_Seasonal"):
            mask_treenames |= np.char.find(tn, "Seasonal_") >= 0
        if opt_params.get("exc_lag"):
            mask_treenames &= np.char.find(tn, "_lag") < 0
            
        Xdtime_tr, _ = self._make_design(Xtr_time[mask_train], t_win)
        Xdtime_te, _ = self._make_design(Xte_time[mask_test], t_win)

        Xd_tr = np.hstack((Xtr_tree[mask_train][:, mask_treenames], Xdtime_tr))
        ytr_tree_opt = ytr_tree_opt[mask_train]
        Xd_te = np.hstack((Xte_tree[mask_test][:, mask_treenames], Xdtime_te))
        
        logger.info(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.info(f"   ytr_tree: n={ytr_tree_opt.size}, mean={ytr_tree_opt.mean():.6f}, std={ytr_tree_opt.std():.6f}")

        if do_transform:
            scaler = StandardScaler().fit(Xd_tr)
            Xd_tr = scaler.transform(Xd_tr)
            Xd_te = scaler.transform(Xd_te)

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

        # Log feature importance
        base = list(timenames[:5])
        T = t_win + 1
        t_feat_names = [
            (f"{base[f]}_t-{t_win - t}" if t_win - t else f"{base[f]}_t")
            for t in range(T) for f in range(5)
        ]
        colnames = tn[mask_treenames].tolist() + t_feat_names
        ModelAnalyzer.print_feature_importance_LGBM(lgbModel=model_lgb, featureColumnNames=colnames, n_feature=5)

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
        res_mask[mask_test] = mask_sel
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train][mask_tr], 
            ytr_tree_low[mask_train][mask_tr], 
            ytr_tree_high[mask_train][mask_tr], 
            ytr_tree_open[mask_train][mask_tr],
            n_grid=10
        )
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        return res_mask, sl_te, tp_te

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
        
        def apply_step(
            mask_train: np.ndarray,
            mask_test: np.ndarray,
            params: dict
        ):
            fs = FilterSamples(
                Xtree_train=Xtr_tree[mask_train],
                ytree_train=ytr_tree[mask_train][:, -1],
                treenames=treenames,
                Xtree_test=Xte_tree[mask_test],
                ytree_test=None,
                meta_train=meta_train.filter(mask_train),
                meta_test=meta_test.filter(mask_test),
                params=params,
            )

            cat_train, cat_test = fs.categorical_masks()

            # Update masks in-place while preserving alignment
            mask_train[mask_train] &= cat_train
            if cat_test is not None:
                mask_test[mask_test] &= cat_test

            return fs, mask_train, mask_test
        
        volat_w = params["volatility_w"]
        volat_dir = params["volatility_dir"]
        volat_q = params["volatility_q"]
        
        volprice_q = params["volprice_q"]
        volprice_w = params["volprice_w"]
        
        predic_w = params["predictability_w"]
        predic_dir = params["predictability_dir"]
        predic_q = params["predictability_q"]
        
        def q_limit(mask_test):
            min_n_tar_daily = 30
            n_dates_test = meta_test.get_column("date").n_unique()
            return min(n_dates_test * min_n_tar_daily / mask_test.sum(), 1.0)
        
        q_l = q_limit(mask_test)
        q = max(predic_q, q_l)
        key = f"FilterSamples_cat_predictability_w{predic_w}_{predic_dir}{q:.2f}"
        step_params = dict(params)
        step_params[key] = True
        logger.debug("Applying first category filter: %s", key)
        _, mask_train, mask_test = apply_step(mask_train, mask_test, step_params)
        logger.debug(f"  After first cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")

        q_l = q_limit(mask_test)
        q = min(volprice_q, 1-q_l)
        key = f"FilterSamples_cat_volumeprice_w{volprice_w}_{q:.2f}"
        step_params = dict(params)
        step_params[key] = True
        logger.debug("Applying second category filter: %s", key)
        _, mask_train, mask_test = apply_step(mask_train, mask_test, step_params)
        logger.debug(f"  After second cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")

        
        q_l = q_limit(mask_test)
        q = min(volat_q, 1-q_l)
        key = f"FilterSamples_cat_volatility_w{volat_w}_{volat_dir}{q:.2f}"
        step_params = dict(params)
        step_params[key] = True
        logger.debug("Applying third category filter: %s", key)
        _, mask_train, mask_test = apply_step(mask_train, mask_test, step_params)
        logger.debug(f"  After third cat filter -> train kept: {100 * mask_train.mean():.2f}% | test kept: {100 * mask_test.mean():.2f}%")
                    
        unique_tickers = meta_test.filter(mask_test).get_column("ticker").unique().to_numpy()
        logger.debug(f"  Precompute -> test unique tickers kept: {unique_tickers.size} | total: {len(meta_test.get_column('ticker').unique())}")
        logger.debug(f"    Tickers: {unique_tickers}")

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid = 5
        )        
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
    def _make_design(self, X, t_win):
        Xw: np.ndarray = X[:, -(t_win+1):, 0:5].copy()
        Xw = (Xw-0.5)*2.0
        
        mask_bad = np.zeros(Xw.shape[0], dtype=bool)
        bound_bad = 1 - np.tanh(1 - 1e-4)
        mask_bad = np.any((Xw[:,:,0:4] <= (-1+bound_bad)) | (Xw[:,:,0:4] >= (1-bound_bad)), axis=(1,2))
        Xw[:,:,0:4] = np.clip(Xw[:,:,0:4], -1+bound_bad, 1-bound_bad)
        
        Xw[:,:,0:4] = np.arctanh(Xw[:,:,0:4]) + 1.0
        
        return Xw.reshape(Xw.shape[0], -1), mask_bad