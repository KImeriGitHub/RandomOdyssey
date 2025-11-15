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
from src.hyperparameterTuning.HelperMetrics import HelperMetrics

import logging
logger = logging.getLogger(__name__)

class StratLGBMOnFiltered(BaseStrategy):
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
        "FilterSamples_cat_highestShareholderEquity_q0.2": False,
        "FilterSamples_cat_volatility_qdown0.02": False,
        "FilterSamples_cat_volatility_qup0.975": False,
        "FilterSamples_cat_predictability_qup0.9": False,
        "FilterSamples_cat_predictability_qdown0.1": False,
    }

    base_params = {
        "LGB_num_boost_round": 950,
        "LGB_lambda_l1": 0.000100,
        "LGB_lambda_l2": 0.009786276249261908,
        "LGB_feature_fraction": 0.20813359498274574,
        "LGB_num_leaves": 191,
        "LGB_max_depth": 9,
        "LGB_learning_rate": 0.008855,
        "LGB_min_data_in_leaf": 350,
        "LGB_min_gain_to_split": 0.10066457576238419,
        "LGB_path_smooth": 0.5935679203578974,
        "LGB_min_sum_hessian_in_leaf": 0.3732876155751053,
        "LGB_max_bin": 850,
    }

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {}
        opt_params["inc_FeatureTA"] = trial.suggest_categorical("inc_FeatureTA", [True, False])
        opt_params["inc_GroupDynamics"] = trial.suggest_categorical("inc_GroupDynamics", [True, False])
        opt_params["inc_Categorical"] = trial.suggest_categorical("inc_Categorical", [True, False])
        opt_params["inc_Financials"] = trial.suggest_categorical("inc_Financials", [True, False])
        opt_params["inc_Mathematical"] = trial.suggest_categorical("inc_Mathematical", [True, False])
        opt_params["inc_Seasonal"] = trial.suggest_categorical("inc_Seasonal", [True, False])
        opt_params["exc_lag"] = trial.suggest_categorical("exc_lag", [True, False])
        
        opt_params["LGB_num_boost_round"]           = trial.suggest_int("LGB_num_boost_round", 25, 300, step=25)
        opt_params["LGB_lambda_l1"]                 = trial.suggest_float("LGB_lambda_l1", 0.00001, 0.05, log=True)
        opt_params["LGB_lambda_l2"]                 = trial.suggest_float("LGB_lambda_l2", 0.0001, 0.02, log=True)
        if opt_params.get("exc_lag"):
            opt_params["LGB_feature_fraction"]          = trial.suggest_float("LGB_feature_fraction", 0.7, 0.9, log=True)
        else:
            opt_params["LGB_feature_fraction"]          = trial.suggest_float("LGB_feature_fraction", 0.01, 0.1, log=True)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 700, 1200, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 4, 30, step=1)
        opt_params["LGB_learning_rate"]             = trial.suggest_float("LGB_learning_rate", 0.0001, 0.005, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 100, 600, step=25)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 0.0001, 0.9, log=True)
        opt_params["LGB_path_smooth"]               = trial.suggest_float("LGB_path_smooth", 0.01, 0.9, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        opt_params["LGB_max_bin"]                   = trial.suggest_int("LGB_max_bin", 100, 800, step=25)
        opt_params["LGB_early_stopping_rounds"]     = opt_params["LGB_num_boost_round"]//10
        
        opt_params["do_transform"] = trial.suggest_categorical("do_transform", [True, False])
        opt_params["val_split"] = trial.suggest_float("val_split", 0.01, 0.05, log=True)

        opt_params["ytree_kind"] = trial.suggest_categorical("ytree_kind", ["last", "abslast", "mean", "max"])

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
        mm: MachineModels = MachineModels(opt_params)
        do_transform =      opt_params["do_transform"]
        val_split =         opt_params["val_split"]
        
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

        if opt_params["ytree_kind"] == "last":
            ytr_tree_opt = ytr_tree[:, -1]
        elif opt_params["ytree_kind"] == "mean":
            ytr_tree_opt = np.mean(ytr_tree, axis=1)
        elif opt_params["ytree_kind"] == "abslast":
            ytr_tree_opt = np.abs(ytr_tree[:, -1])
        elif opt_params["ytree_kind"] == "max":
            ytr_tree_opt = np.max(ytr_tree, axis=1)
            
        logger.info(f"  Before filtering: tr {Xtr_tree.shape}, te {Xte_tree.shape}")

        Xd_tr = Xtr_tree[:, mask_treenames]
        Xd_te = Xte_tree[:, mask_treenames]
        
        logger.info(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.info(f"   ytr_tree: n={ytr_tree_opt.size}, mean={ytr_tree_opt.mean():.6f}, std={ytr_tree_opt.std():.6f}")


        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree, 
            ytr_tree_low, 
            ytr_tree_high, 
            ytr_tree_open
        )
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        if do_transform:
            scaler = StandardScaler().fit(Xd_tr)
            Xd_tr = scaler.transform(Xd_tr)
            Xd_te = scaler.transform(Xd_te)

        sam_split = int(Xd_tr.shape[0] * (1 - val_split))
        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xd_tr[:sam_split],
                y_train=ytr_tree_opt[:sam_split],
                X_test=Xd_tr[sam_split:],
                y_test=ytr_tree_opt[sam_split:],
            )
        except Exception as e:
            logger.disabled = False
            logger.warning(f"  LGB failed: {e}")
            return 1.0
        finally:
            logger.disabled = False

        # LGB Predictions
        best_iter = getattr(model_lgb, "best_iteration", None)
        y_test_pred_masked  = model_lgb.predict(Xd_te, num_iteration=best_iter)
        logger.info(f"  Test RMSE LGBM: {info['best_score']:.4f}")
        
        m = 5
        meta_pl_filtered = (
            meta_test.with_columns(
                pl.Series("prediction_ratio", y_test_pred_masked)
            )
            .sort(["date", "prediction_ratio"], descending=[False, True])
            .with_columns(
                pl.col("prediction_ratio")
                .rank(method="random", descending=True)
                .over("date")
                .alias("prediction_rank")
            )
        )
        mask_te = meta_pl_filtered["prediction_rank"].to_numpy() <= m

        return mask_te, sl_te, tp_te

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
            ytree_train=ytr_tree[:,-1],
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

        do_transform =      True
        tree_n_max =        1
        min_n_tar =         75
        top_n_max =         375
        
        cat_ratio =         0.05
        params[f"FilterSamples_cat_volatility_qdown{cat_ratio:.2f}"] = False
        params[f"FilterSamples_cat_volatility_qup{(1-cat_ratio):.2f}"] = True
        params["LGB_num_boost_round"]           = 1 #trial.suggest_int("LGB_num_boost_round", 40, 60, step=1)
        params["LGB_lambda_l1"]                 = 0.169225
        params["LGB_lambda_l2"]                 = 0.000651
        params["apply_feature_fraction"]        = True
        if params["apply_feature_fraction"]:
            params["LGB_feature_fraction"]      = 0.950331
        params["LGB_num_leaves"]                = 1525
        params["LGB_max_depth"]                 = 22
        params["LGB_learning_rate"]             = 0.1 #trial.suggest_float("LGB_learning_rate", 1e-4, 2e-0, log=True)
        params["LGB_min_data_in_leaf"]          = 270
        params["LGB_min_gain_to_split"]         = 5.337715e-07
        params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 1e-2, 5e-1, log=True)
        params["LGB_min_sum_hessian_in_leaf"]   = 0.207050
        params["LGB_max_bin"]                   = 290

        mm: MachineModels = MachineModels(params)
            
        logger.info(f"  Before filtering: tr {Xtr_tree.shape}, te {Xte_tree.shape}")
            
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
            params=params,
        )

        cat_train, cat_test = fs_pre.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test

        logger.info(f"  After filtering: tr {(int(mask_train.sum()), Xtr_tree.shape[1])}, te {(int(mask_test.sum()), Xte_tree.shape[1])}")

        tn = np.asarray(treenames, dtype=str)
        mask_treenames = np.zeros(len(treenames), dtype=bool)
        mask_treenames |= np.char.find(tn, "FeatureGroup_") >= 0
        mask_treenames |= np.char.find(tn, "Seasonal_") >= 0

        Xd_tr, ytr_tree = Xtr_tree[mask_train][:, mask_treenames], ytr_tree[mask_train]
        Xd_te = Xte_tree[mask_test][:, mask_treenames]

        ytr_tree_opt = np.abs(ytr_tree[:, -1])

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

        res_tr_mask = np.zeros(Xtr_tree.shape[0], dtype=bool)
        res_tr_mask[mask_train] = mask_tr
        res_te_mask = np.zeros(Xte_tree.shape[0], dtype=bool)
        res_te_mask[mask_test] = mask_sel

        logger.info(f"  Precompute selected: {res_tr_mask.sum()} train samples and {res_te_mask.sum()} test samples.")

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_tr], 
            ytr_tree_low[mask_train][mask_tr], 
            ytr_tree_high[mask_train][mask_tr], 
            ytr_tree_open[mask_train][mask_tr],
            n_grid=7
        )
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        return res_tr_mask, res_te_mask, sl_tr, sl_te, tp_tr, tp_te

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
