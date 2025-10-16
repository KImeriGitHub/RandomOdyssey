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
    val_split = 0.05
    expected_load_params = {
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }
    precompute_params = {
        "FilterSamples_q_up": 0.96,
        "FilterSamples_method": "taylor",
        "FilterSamples_days_to_train_end": 4,

        "FilterSamples_cat_over20.0": True,
        "FilterSamples_cat_under2000.0": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_doubleFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.5": True,
        
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

        "FilterSamples_taylor_horizon_days": 6,
        "FilterSamples_taylor_roll_window_days": 6,
        "FilterSamples_taylor_weight_slope": 0.85,
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
        opt_params["LGB_num_boost_round"]           = trial.suggest_int("LGB_num_boost_round", 75, 200, step=5)
        opt_params["LGB_lambda_l1"]                 = trial.suggest_float("LGB_lambda_l1", 0.0001, 0.005, log=True)
        opt_params["LGB_lambda_l2"]                 = trial.suggest_float("LGB_lambda_l2", 0.001, 0.05, log=True)
        opt_params["LGB_feature_fraction"]          = trial.suggest_float("LGB_feature_fraction", 0.8, 0.99)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 800, 1500, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 4, 17, step=1)
        opt_params["LGB_learning_rate"]             = trial.suggest_float("LGB_learning_rate", 0.0001, 2.0, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 200, 600, step=25)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 0.0001, 0.02, log=True)
        opt_params["LGB_path_smooth"]               = trial.suggest_float("LGB_path_smooth", 0.01, 0.9, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.0001, 0.005, log=True)
        opt_params["LGB_max_bin"]                   = trial.suggest_int("LGB_max_bin", 400, 800, step=25)
        opt_params["LGB_early_stopping_rounds"]     = opt_params["LGB_num_boost_round"]//10

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
        mm: MachineModels = MachineModels(opt_params)

        Xd_tr, ytr_tree = Xtr_tree, ytr_tree
        Xd_te, yte_tree = Xte_tree, yte_tree

        sl_val, tp_val = HelperFunctions.optimal_sl_tp(ytr_tree)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        logger.debug(f"   ytr_tree: n={ytr_tree.size}, mean={ytr_tree.mean():.6f}, std={ytr_tree.std():.6f}")

        if not opt_params.get("LoadupSamples_tree_scaling_standard", False):
            scaler = StandardScaler().fit(Xd_tr)
            Xd_tr = scaler.transform(Xd_tr)
            Xd_te = scaler.transform(Xd_te)

        sam_split = int(Xd_tr.shape[0] * (1 - self.val_split))
        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xd_tr[:sam_split],
                y_train=ytr_tree[:sam_split],
                X_test=Xd_tr[sam_split:],
                y_test=ytr_tree[sam_split:],
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

        ### MAIN FILTERING
        fs = FilterSamples(
            Xtree_train = Xtr_tree[mask_train], 
            ytree_train = ytr_tree[mask_train], 
            treenames   = treenames,
            Xtree_test  = Xte_tree[mask_test],
            ytree_test  = None,
            meta_train  = meta_train.filter(pl.Series(mask_train)), 
            meta_test   = meta_test.filter(pl.Series(mask_test)), 
            params      = params,
        )

        if params["FilterSamples_method"] == "taylor":
            mask_train, mask_test = fs.taylor_feature_masks()
        if params["FilterSamples_method"] == "lincomb":
            mask_train, mask_test = fs.lincomb_masks()

        score_train = fs.evaluate_mask(
            mask_train, 
            meta_train.filter(pl.Series(mask_train))['date'], 
            ytr_tree[mask_train][:, -1]
        )
        logger.info(f"  Filtering Score (train) = {score_train}")

        mask_train[mask_train] = mask_train
        if mask_test is not None:
            mask_test[mask_test] = mask_test

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
