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
        mm: MachineModels = MachineModels(opt_params)

        Xd_tr, ytr_tree = Xtr_tree, ytr_tree
        Xd_te, yte_tree = Xte_tree, yte_tree

        logger.debug(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.debug(f"   ytr_tree: n={ytr_tree.size}, mean={ytr_tree.mean():.6f}, std={ytr_tree.std():.6f}")

        if not self.base_params.get("LoadupSamples_tree_scaling_standard", False):
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
            ).filter(pl.col("prediction_rank") <= m)
        )
        agg_exprs = [
            pl.col("prediction_ratio").max().alias("max_pred"),  # this is also .first()
            pl.col("prediction_ratio").log().mean().exp().alias("mean_pred"),
            pl.col("target_ratio").log().mean().exp().alias("mean_res"),
            pl.col("target_ratio")
                .sort_by(pl.col("prediction_ratio"), descending=True)
                .first()
                .alias("top_res"),
            pl.len().alias("n_entries"),
        ]
        test_df_perdate = meta_pl_filtered.group_by("date").agg(agg_exprs).sort("date")
        
        if test_df_perdate.height == 0:
            pred_meanlast = pred_toplast = res_meanlast = res_toplast = predmeanmean = 1.0
        else:
            predmeanmean  = test_df_perdate["mean_pred"].mean()
            pred_meanlast = test_df_perdate["mean_pred"].item(-1)
            pred_toplast  = test_df_perdate["max_pred"].item(-1)
            res_meanlast  = test_df_perdate["mean_res"].item(-1)
            res_toplast   = test_df_perdate["top_res"].item(-1)
        res_sum_n = int(test_df_perdate["n_entries"].sum())
        
        score = res_meanlast

        # Final  Analysis
        logger.info(f"  Final top last prediction ratio: {pred_toplast:.4f}")
        logger.info(f"  Final last mean prediction ratio: {pred_meanlast:.4f}")
        logger.info(f"  Final top last P/L Ratio: {res_toplast:.4f}")
        logger.info(f"  Final mean last P/L Ratio: {res_meanlast:.4f}")
        logger.info(f"  Number of entries: {res_sum_n}")
        logger.info(f"  Score value: {score:.4f}")

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

        ### MAIN FILTERING
        fs = FilterSamples(
            Xtree_train = Xtr_tree[cat_mask_train], 
            ytree_train = ytr_tree[cat_mask_train], 
            treenames   = treenames,
            Xtree_test  = Xte_tree[cat_mask_test],
            ytree_test  = yte_tree[cat_mask_test],
            meta_train  = meta_train.filter(pl.Series(cat_mask_train)), 
            meta_test   = meta_test.filter(pl.Series(cat_mask_test)), 
            params      = params,
        )

        if params["FilterSamples_method"] == "taylor":
            mask_train, mask_test = fs.taylor_feature_masks()
        if params["FilterSamples_method"] == "lincomb":
            mask_train, mask_test = fs.lincomb_masks()

        score_train = fs.evaluate_mask(mask_train, 
            meta_train.filter(pl.Series(cat_mask_train))['date'], ytr_tree[cat_mask_train])
        score_test  = fs.evaluate_mask(mask_test,  
            meta_test.filter(pl.Series(cat_mask_test))['date'], yte_tree[cat_mask_test])
        logger.info(f"  Filtering Score (train) = {score_train}")
        logger.info(f"  Filtering Score (test)  = {score_test}")

        cat_mask_train[cat_mask_train] = mask_train
        if mask_test is not None:
            cat_mask_test[cat_mask_test] = mask_test

        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * cat_mask_train.mean(),
            100 * cat_mask_test.mean(),
        )

        return cat_mask_train, cat_mask_test

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
