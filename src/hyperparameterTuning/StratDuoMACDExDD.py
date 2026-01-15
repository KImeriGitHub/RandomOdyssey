import numpy as np
import optuna
import polars as pl
import re

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.hyperparameterTuning.HelperFunctions import HelperFunctions
from src.hyperparameterTuning.HelperSLTP import HelperSLTP
from sklearn.ensemble import RandomForestRegressor

from src.predictionModule.MachineModels import MachineModels
from src.predictionModule.WeightSamples import WeightSamples
from src.predictionModule.FilterSamples import FilterSamples

from scipy.optimize import differential_evolution

import logging
logger = logging.getLogger(__name__)

class StratDuoMACDExDD(BaseStrategy):
    # Note Xtr_time needs the features
    #        "FeatureLSTM_AdjClose" at index 0,
    #        "FeatureLSTM_AdjOpen" at index 1,
    #        "FeatureLSTM_AdjHigh" at index 2,
    #        "FeatureLSTM_AdjLow" at index 3,
    
    expected_load_params = {
        "idxAfterPrediction": 5,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "FilterSamples_days_to_train_end": 255 * 6,
        
        "doubledrop_thr_tr" : 0.82,
        
        "q_len_macd": 0.20,
        "macd_n_fast": 7,
        "macd_n_slow": 2,
        "macd_n_sgnl": 1,
    }

    base_params = {
    }

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {}
        
        # -- DD ---
        opt_params["DD_catsample_max_lag"] = 0 #trial.suggest_int("DD_catsample_max_lag", 0, 155, step=5) # Note: try 100 at some point
        opt_params["DD_max_features_colinsampling"] = trial.suggest_int("DD_max_features_colinsampling", 30, 250, step=5)
        opt_params["DD_threshold_colin_sampling"] = trial.suggest_float("DD_threshold_colin_sampling", 0.08, 0.13, log=True)
        
        opt_params["DD_keep_perday_rf"]               = 2
        
        opt_params["DD_LGB_num_boost_round"]           = trial.suggest_categorical("DD_LGB_num_boost_round", [300, 700, 1500, 2500])
        opt_params["DD_LGB_lambda_l1"]                 = 0.0005 #trial.suggest_float("DD_LGB_lambda_l1", 0.000001, 0.001, log=True)
        opt_params["DD_LGB_lambda_l2"]                 = 0.0 #trial.suggest_float("DD_LGB_lambda_l2", 0.000001, 0.0001, log=True)
        opt_params["DD_LGB_feature_fraction"]          = 1.0 #trial.suggest_float("DD_LGB_feature_fraction", 0.8, 1.0)
        opt_params["DD_LGB_num_leaves"]                = trial.suggest_int("DD_LGB_num_leaves", 150, 850, step=50)
        opt_params["DD_LGB_max_depth"]                 = trial.suggest_int("DD_LGB_max_depth", 10, 35, step=1)
        opt_params["DD_LGB_learning_rate"]             = trial.suggest_float("DD_LGB_learning_rate", 0.01, 0.15, log=True)
        opt_params["DD_LGB_min_data_in_leaf"]          = trial.suggest_int("DD_LGB_min_data_in_leaf", 400, 2400, step=10)
        opt_params["DD_LGB_min_gain_to_split"]         = trial.suggest_float("DD_LGB_min_gain_to_split", 0.00001, 0.01, log=True)
        opt_params["DD_LGB_path_smooth"]               = 0.210775 #trial.suggest_float("DD_LGB_path_smooth", 0.1, 0.9, log=True)
        opt_params["DD_LGB_min_sum_hessian_in_leaf"]   = 0.213247 #trial.suggest_float("DD_LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        opt_params["DD_LGB_max_bin"]                   = 250 #trial.suggest_int("DD_LGB_max_bin", 100, 800, step=25)
        opt_params["DD_LGB_early_stopping_rounds"]     = opt_params["DD_LGB_num_boost_round"]//10
        
        # -- MACD ---
        opt_params["MACD_catsample_max_lag"] = trial.suggest_int("MACD_catsample_max_lag", 0, 155, step=5) # Note: try 100 at some point
        opt_params["MACD_max_features_colinsampling"] = trial.suggest_int("MACD_max_features_colinsampling", 20, 250, step=5)
        opt_params["MACD_threshold_colin_sampling"] = trial.suggest_float("MACD_threshold_colin_sampling", 0.08, 0.13, log=True)
        
        opt_params["MACD_keep_perday_rf"]               = 2
        
        opt_params["MACD_LGB_num_boost_round"]           = trial.suggest_categorical("MACD_LGB_num_boost_round", [300, 700, 1500, 2500])
        opt_params["MACD_LGB_lambda_l1"]                 = 0.0005 #trial.suggest_float("MACD_LGB_lambda_l1", 0.000001, 0.001, log=True)
        opt_params["MACD_LGB_lambda_l2"]                 = 0.0 #trial.suggest_float("MACD_LGB_lambda_l2", 0.000001, 0.0001, log=True)
        opt_params["MACD_LGB_feature_fraction"]          = 1.0 #trial.suggest_float("MACD_LGB_feature_fraction", 0.8, 1.0)
        opt_params["MACD_LGB_num_leaves"]                = trial.suggest_int("MACD_LGB_num_leaves", 250, 1500, step=50)
        opt_params["MACD_LGB_max_depth"]                 = trial.suggest_int("MACD_LGB_max_depth", 20, 100, step=5)
        opt_params["MACD_LGB_learning_rate"]             = trial.suggest_float("MACD_LGB_learning_rate", 0.01, 0.15, log=True)
        opt_params["MACD_LGB_min_data_in_leaf"]          = trial.suggest_int("MACD_LGB_min_data_in_leaf", 400, 2400, step=50)
        opt_params["MACD_LGB_min_gain_to_split"]         = trial.suggest_float("MACD_LGB_min_gain_to_split", 0.00001, 0.01, log=True)
        opt_params["MACD_LGB_path_smooth"]               = 0.2 #trial.suggest_float("MACD_LGB_path_smooth", 0.1, 0.9, log=True)
        opt_params["MACD_LGB_min_sum_hessian_in_leaf"]   = 0.2 #trial.suggest_float("MACD_LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        opt_params["MACD_LGB_max_bin"]                   = 250 #trial.suggest_int("MACD_LGB_max_bin", 100, 800, step=25)
        opt_params["MACD_LGB_early_stopping_rounds"]     = opt_params["MACD_LGB_num_boost_round"]//10
        
        
        opt_params["DD_inc_weight_features"]   = trial.suggest_categorical("DD_inc_weight_features", [True, False])
        if opt_params["DD_inc_weight_features"]:
            opt_params["DD_weight_wndw_ratio"]     = trial.suggest_float("DD_weight_wndw_ratio", 0.01, 0.3, log=True)
            opt_params["DD_min_weight"]            = trial.suggest_float("DD_min_weight", 0.1, 2.0, step=0.1)
            opt_params["DD_trunc_weight_features"] = trial.suggest_int("DD_trunc_weight_features", 1, 4)
            
        opt_params["MACD_inc_weight_features"]   = trial.suggest_categorical("MACD_inc_weight_features", [True, False])
        if opt_params["MACD_inc_weight_features"]:
            opt_params["MACD_weight_wndw_ratio"]     = trial.suggest_float("MACD_weight_wndw_ratio", 0.01, 0.3, log=True)
            opt_params["MACD_min_weight"]            = trial.suggest_float("MACD_min_weight", 0.1, 2.0, step=0.1)
            opt_params["MACD_trunc_weight_features"] = trial.suggest_int("MACD_trunc_weight_features", 1, 4)
            
        # --- SLTP gate ---
        opt_params["DD_sl_val"] = trial.suggest_float("DD_sl_val", 0.86, 0.93)
        opt_params["DD_tp_val"] = trial.suggest_float("DD_tp_val", 1.2, 1.5)
        
        opt_params["MACD_sl_val"] = trial.suggest_float("MACD_sl_val", 0.86, 0.93)
        opt_params["MACD_tp_val"] = trial.suggest_float("MACD_tp_val", 1.2, 1.5)
        
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
    ) -> tuple[np.ndarray, ...]:
        ytr_opt = ytr_tree[:, -1]
        tn = np.asarray(treenames, str)
        
        # --- dd mask ---
        close_tr = Xtr_time[:, :, 0].astype(np.float64, copy=False)
        close_te = Xte_time[:, :, 0].astype(np.float64, copy=False)
        
        mask_tr_dd = (close_tr[:, -1] < close_tr[:, -2]) & (close_tr[:, -2] < close_tr[:, -3])
        mask_te_dd = (close_te[:, -1] < close_te[:, -2]) & (close_te[:, -2] < close_te[:, -3])

        # --- feature and weight selection ---
        feat_tr_dd_res, feat_te_dd_res, weights_dd, mask_tn_dd = self._select_feature_subset_and_weights(
            prefix="DD",
            opt_params=opt_params,
            Xtr_tree=Xtr_tree,
            Xte_tree=Xte_tree,
            ytr_opt=ytr_opt,
            treenames=treenames,
            tn=tn,
            mask_tr=mask_tr_dd,
            mask_te=mask_te_dd
        )
          
        feat_tr_macd_res, feat_te_macd_res, weights_macd, mask_tn_macd = self._select_feature_subset_and_weights(
            prefix="MACD",
            opt_params=opt_params,
            Xtr_tree=Xtr_tree,
            Xte_tree=Xte_tree,
            ytr_opt=ytr_opt,
            treenames=treenames,
            tn=tn,
            mask_tr=~mask_tr_dd,
            mask_te=~mask_te_dd
        )
        
        # --- LGBModels ---
        _, score_tr_DD_sub, score_te_DD_sub, keep_tr_DD_sub, keep_te_DD_sub = self._rf_rank_keep_perday(
            prefix="DD",
            opt_params=opt_params,
            feat_tr=feat_tr_dd_res,
            feat_te=feat_te_dd_res,
            ytr=ytr_opt[mask_tr_dd],
            weights=weights_dd,
            meta_train=meta_train.filter(pl.Series(mask_tr_dd)),
            meta_test=meta_test.filter(pl.Series(mask_te_dd)),
        )
        
        _, score_tr_MACD_sub, score_te_MACD_sub, keep_tr_MACD_sub, keep_te_MACD_sub = self._rf_rank_keep_perday(
            prefix="MACD",
            opt_params=opt_params,
            feat_tr=feat_tr_macd_res,
            feat_te=feat_te_macd_res,
            ytr=ytr_opt[~mask_tr_dd],
            weights=weights_macd,
            meta_train=meta_train.filter(pl.Series(~mask_tr_dd)),
            meta_test=meta_test.filter(pl.Series(~mask_te_dd)),
        )
        
        # --- combining ---
        mask_train = np.zeros(Xtr_tree.shape[0], dtype=bool)
        mask_test  = np.zeros(Xte_tree.shape[0], dtype=bool)

        mask_train[mask_tr_dd] = keep_tr_DD_sub
        mask_test[mask_te_dd]  = keep_te_DD_sub
        mask_train[~mask_tr_dd] = keep_tr_MACD_sub
        mask_test[~mask_te_dd]  = keep_te_MACD_sub

        score_tr = np.zeros(Xtr_tree.shape[0], dtype=np.float32)
        score_te = np.zeros(Xte_tree.shape[0], dtype=np.float32)
        
        score_tr[mask_tr_dd] = score_tr_DD_sub
        score_tr[~mask_tr_dd] = score_tr_MACD_sub
        score_te[mask_te_dd] = score_te_DD_sub
        score_te[~mask_te_dd] = score_te_MACD_sub
                
        # --- SL/TP matrices ---
        DD_sl_val = opt_params["DD_sl_val"]
        DD_tp_val = opt_params["DD_tp_val"]
        MACD_sl_val = opt_params["MACD_sl_val"]
        MACD_tp_val = opt_params["MACD_tp_val"]
        
        sl_tr = np.ones((Xtr_tree.shape[0], ytr_tree.shape[1]), dtype=float)
        sl_te = np.ones((Xte_tree.shape[0], ytr_tree.shape[1]), dtype=float)
        tp_tr = np.ones((Xtr_tree.shape[0], ytr_tree.shape[1]), dtype=float)
        tp_te = np.ones((Xte_tree.shape[0], ytr_tree.shape[1]), dtype=float)
        
        sl_tr[mask_tr_dd]  = sl_tr[mask_tr_dd]  * DD_sl_val
        sl_tr[~mask_tr_dd] = sl_tr[~mask_tr_dd] * MACD_sl_val
        sl_te[mask_te_dd]  = sl_te[mask_te_dd]  * DD_sl_val
        sl_te[~mask_te_dd] = sl_te[~mask_te_dd] * MACD_sl_val
        
        tp_tr[mask_tr_dd]  = tp_tr[mask_tr_dd]  * DD_tp_val
        tp_tr[~mask_tr_dd] = tp_tr[~mask_tr_dd] * MACD_tp_val
        tp_te[mask_te_dd]  = tp_te[mask_te_dd]  * DD_tp_val
        tp_te[~mask_te_dd] = tp_te[~mask_te_dd] * MACD_tp_val
        
        # --- logging ---
        logger.info(
            "  Final masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        
        logger.info(f"  Final SLTP -> sl tr {sl_tr[:,0].mean()} | tp tr: {tp_tr[:,0].mean()}")

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te, score_tr, score_te

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
        
        assert timenames.index("FeatureLSTM_AdjClose") == 0, "FeatureLSTM_AdjClose is required in X*_time for MACD computation."
        assert timenames.index("FeatureLSTM_AdjHigh") == 2, "FeatureLSTM_AdjHigh is required in X*_time for ATR computation."
        assert timenames.index("FeatureLSTM_AdjLow") == 3, "FeatureLSTM_AdjLow is required in X*_time for ATR computation."
        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
        
        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree[:,-1],
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=None,
            meta_train=meta_train,
            meta_test=meta_test,
            params=self.precompute_params,
        )
            
        mask_recent_training = fs.get_recent_training_mask()
        
        # --- doubledrop masks ---
        close_tr = Xtr_time[mask_train][:, :, 0].astype(np.float64, copy=False)
        close_te = Xte_time[mask_test][:, :, 0].astype(np.float64, copy=False)
        
        mask_tr_dd = (close_tr[:, -1] < close_tr[:, -2]) & (close_tr[:, -2] < close_tr[:, -3])
        mask_te_dd = (close_te[:, -1] < close_te[:, -2]) & (close_te[:, -2] < close_te[:, -3])
        
        feat_tr = close_tr[:, -1] / close_tr[:, -3]
        feat_te = close_te[:, -1] / close_te[:, -3]
        
        qthr = self.precompute_params.get("doubledrop_thr_tr")
        mask_tr_dd &= feat_tr > qthr
        mask_te_dd &= feat_te > qthr
        
        logger.info(f"  DoubleDrop -> quantile thr value: {qthr}")
        
        # --- macd masks ---
        _, _, mask_tr_macd, mask_te_macd = self._compute_macd_feat(
            Xtr_time,
            Xte_time,
            ytr_tree[:, -1],
            self.precompute_params,
        )
        
        # --- combine masks ---
        mask_train = mask_train & mask_recent_training & (mask_tr_dd | mask_tr_macd)
        mask_test = mask_test & (mask_te_dd | mask_te_macd)
        
        # -- log info ---
        n_dates_tr = meta_train.get_column('date').n_unique()
        n_dates_te = meta_test.get_column('date').n_unique()
        
        n_dates_masked_tr = meta_train.filter(pl.Series(mask_train)).get_column('date').n_unique()
        n_dates_masked_te = meta_test.filter(pl.Series(mask_test)).get_column('date').n_unique()
        
        logger.info(f"  Dates kept -> train: {n_dates_masked_tr / n_dates_tr} | test: {n_dates_masked_te / n_dates_te}")

        sl_val, tp_val = 0.88, 2.0
        sl_tr = sl_val * np.ones((Xtr_tree.shape[0], ytr_tree.shape[1]), dtype=float)
        sl_te = sl_val * np.ones((Xte_tree.shape[0], ytr_tree.shape[1]), dtype=float)
        tp_tr = tp_val * np.ones((Xtr_tree.shape[0], ytr_tree.shape[1]), dtype=float)
        tp_te = tp_val * np.ones((Xte_tree.shape[0], ytr_tree.shape[1]), dtype=float)
        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(f"  Precompute -> sl {sl_val} | tp: {tp_val}")

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te
    
    #------------------------------------------------------------------
    # Helper methods
    #------------------------------------------------------------------
    def _rf_rank_keep_perday(
        self,
        *,
        prefix: str,
        opt_params: dict,
        feat_tr: np.ndarray,
        feat_te: np.ndarray,
        ytr: np.ndarray,
        weights,
        meta_train: pl.DataFrame,
        meta_test: pl.DataFrame,
    ):
        """
        Train LGB on (feat_tr, ytr) with optional weights, predict train/test,
        then keep top-k per day based on predicted score ranks.
        Returns: (model, scores_tr, scores_te, mask_tr_keep, mask_te_keep)
        """
        lgb_params = {
            "LGB_num_boost_round": opt_params[f"{prefix}_LGB_num_boost_round"],
            "LGB_lambda_l1": opt_params[f"{prefix}_LGB_lambda_l1"],
            "LGB_lambda_l2": opt_params[f"{prefix}_LGB_lambda_l2"],
            "LGB_feature_fraction": opt_params[f"{prefix}_LGB_feature_fraction"],
            "LGB_num_leaves": opt_params[f"{prefix}_LGB_num_leaves"],
            "LGB_max_depth": opt_params[f"{prefix}_LGB_max_depth"],
            "LGB_learning_rate": opt_params[f"{prefix}_LGB_learning_rate"],
            "LGB_min_data_in_leaf": opt_params[f"{prefix}_LGB_min_data_in_leaf"],
            "LGB_min_gain_to_split": opt_params[f"{prefix}_LGB_min_gain_to_split"],
            "LGB_path_smooth": opt_params[f"{prefix}_LGB_path_smooth"],
            "LGB_min_sum_hessian_in_leaf": opt_params[f"{prefix}_LGB_min_sum_hessian_in_leaf"],
            "LGB_max_bin": opt_params[f"{prefix}_LGB_max_bin"],
            "LGB_early_stopping_rounds": opt_params[f"{prefix}_LGB_early_stopping_rounds"],
            "LGB_bagging_fraction": 1.0,
        }

        mm: MachineModels = MachineModels(params=lgb_params, use_default_params = False)
        model, _ = mm.run_LGB(feat_tr, ytr, weights=weights)

        scores_tr = model.predict(feat_tr, num_iteration=model.best_iteration)
        scores_te = model.predict(feat_te, num_iteration=model.best_iteration)

        keep_perday = opt_params[f"{prefix}_keep_perday_rf"]

        mask_tr_keep = (
            meta_train
            .with_columns(pl.Series(scores_tr).alias("score"))
            .with_columns(
                pl.col("score")
                .rank(method="random", descending=True)
                .over("date")
                .alias("scores_rank")
            )
            .select(pl.col("scores_rank") <= keep_perday)
            .to_numpy()
            .flatten()
        )

        mask_te_keep = (
            meta_test
            .with_columns(pl.Series(scores_te).alias("score"))
            .with_columns(
                pl.col("score")
                .rank(method="random", descending=True)
                .over("date")
                .alias("scores_rank")
            )
            .select(pl.col("scores_rank") <= keep_perday)
            .to_numpy()
            .flatten()
        )

        return model, scores_tr, scores_te, mask_tr_keep, mask_te_keep

    def _select_feature_subset_and_weights(
        self,
        *,
        prefix: str,
        opt_params: dict,
        Xtr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        ytr_opt: np.ndarray,
        treenames: np.ndarray,
        tn: np.ndarray,
        n_bin: int = 15,
        mask_tr: np.ndarray,
        mask_te: np.ndarray,
    ):
        """
        Applies the same pipeline you have for DD:
        1) categorical subsampling of treenames (lag/non-colin)
        2) colinear sampling based on target correlation
        3) (optional) weight feature generation + WeightSamples weights

        Returns
        -------
        feat_tr_res, feat_te_res, weights, mask_tn
        """
        # --- feature selection params ---
        catsample_max_lag = opt_params[f"{prefix}_catsample_max_lag"]
        m_features = opt_params[f"{prefix}_max_features_colinsampling"]
        thr_colin = opt_params[f"{prefix}_threshold_colin_sampling"]

        # --- build treename mask ---
        mask_tn = np.ones(tn.shape[0], dtype=bool)

        # Categorical subsampling of treenames
        mask_tn_lag, mask_tn_noncolin = self.catsample_treenames(
            treenames,
            max_lag=catsample_max_lag,
        )
        mask_tn &= (mask_tn_lag | mask_tn_noncolin)

        # Colinear sampling based on target correlation
        mask_tn_colin_input = mask_tn_lag & ~mask_tn_noncolin
        if mask_tn_colin_input.any():
            mask_tn_colin_keep = HelperFunctions.colinearity_treenames_mask(
                Xtr_tree[mask_tr][:, mask_tn_colin_input],
                ytr_opt[mask_tr],
                treenames=tn[mask_tn_colin_input],
                m_features=m_features,
                thr_crosscorr=thr_colin,
            )
            mask_tn[mask_tn_colin_input] = mask_tn_colin_keep

        feat_tr_res = Xtr_tree[mask_tr][:, mask_tn]
        feat_te_res = Xte_tree[mask_te][:, mask_tn]

        # --- weight features ---
        if opt_params[f"{prefix}_inc_weight_features"]:
            trunc_weights = opt_params.get(f"{prefix}_trunc_weight_features", 4)

            weightfeat_tr, weightfeat_te, wfeat_names = self._weight_features(
                Xtr_tree[mask_tr], Xte_tree[mask_te], treenames, trunc=trunc_weights
            )

            ws: WeightSamples = WeightSamples(
                feat_train=weightfeat_tr,
                y_train=ytr_opt[mask_tr],
                treenames=wfeat_names,
                feat_test=weightfeat_te,
            )
            weights = ws.establish_weights(
                n_bin=n_bin,
                min_bd=opt_params[f"{prefix}_min_weight"],
                wndw_ratio=opt_params[f"{prefix}_weight_wndw_ratio"],
            )
        else:
            weights = None

        return feat_tr_res, feat_te_res, weights, mask_tn
    
    def _weight_features(
        self,
        Xtr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        treenames: list[str],
        trunc: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        trunc = max(1, min(4, trunc))

        feats = [
            "MathFeature_Return_log",
            "FeatureGroup_WeightedIndexPct",
            "FeatureTA_volatility_ui",
            "MathFeature_Drawdown_MH2",
        ][trunc-1:trunc]

        idx = [treenames.index(f) for f in feats]
        return Xtr_tree[:, idx], Xte_tree[:, idx], feats
    
    def catsample_treenames(
            self,
            treenames: np.ndarray,
            max_lag: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            tn = np.asarray(treenames, str)
            
            # max lag sampling
            mask_tn_lag = np.array(
                [(m:=re.search(r'_lag_m(\d+)', s)) is None or int(m.group(1)) <= max_lag
                    for s in tn]
            )
            
            # not usable for colin
            mask_tn_non_colin = np.zeros(tn.shape[0], dtype=bool)
            mask_tn_lag |= np.char.find(tn, "Seasonal_year") >= 0
            mask_tn_lag |= np.char.find(tn, "Category_other") >= 0
            return mask_tn_lag, mask_tn_non_colin
                
                    
    def _get_ohlc_from_Xtime(
        self,
        X_time: np.ndarray,      # (n_samples, n_steps, n_features)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts OHLC arrays from X_time.
        Uses features:
            0: AdjClose
            1: AdjOpen
            2: AdjHigh
            3: AdjLow
        """
        close = X_time[:, :, 0].astype(np.float64, copy=False)
        open_ = X_time[:, :, 1].astype(np.float64, copy=False)
        high  = X_time[:, :, 2].astype(np.float64, copy=False)
        low   = X_time[:, :, 3].astype(np.float64, copy=False)
        
        return open_, high, low, close
        
    def _compute_macd_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        ytr_vals: np.ndarray,      # (n_samples,)
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # --- parameters ---
        q_len_macd = opt_params["q_len_macd"]
        num_fast = opt_params["macd_n_fast"]
        num_slow = opt_params["macd_n_slow"]
        num_sgnl = opt_params["macd_n_sgnl"]

        # --- Extract close ---
        _, _, _, close_tr = self._get_ohlc_from_Xtime(Xtr_time)
        _, _, _, close_te = self._get_ohlc_from_Xtime(Xte_time)
        
        def geom_vector(a, n, eps=1e-12):
            """
            v = [a^(n-1), a^(n-2), ..., a^0] * (1-a)/(1-a^n)
            """
            a = float(a)
            if a <= -1.0:
                a = -1.0 + eps
            n = int(n)
            if np.isclose(a, 1.0):
                return np.full(n, 1.0/n)
            p = np.arange(n-1, -1, -1)
            scale = (1.0 - a) / (1.0 - a**n)
            return (a**p) * scale
        
        def geom_upper_tri(a, n, eps=1e-12):
            """
            Notes:
                - If a is >1 or <-1, clip to +/-1
            Desc:
                U is n x n upper triangular.
                Column j (0-based) is:
                [a^j, a^(j-1), ..., a^0, 0, ..., 0]^T * (1-a)/(1-a^(j+1))
                So last column (j=n-1) matches the vector in geom_vector.
            """
            a = float(np.clip(a, -1.0, 1.0))
            if a <= -1.0:
                a = -1.0 + eps
            n = int(n)
            if n <= 0:
                return np.zeros((0, 0), dtype=float)

            i = np.arange(n)[:, None]
            j = np.arange(n)[None, :]
            is_upper = (i <= j)

            k = np.arange(1, n + 1)  # column index + 1

            if np.isclose(a, 1.0):
                # Avoid computing (1 - a**k) when a~1
                scales = 1.0 / k
                powers = np.where(is_upper, 1.0, 0.0)
            else:
                exponents = (j - i)
                powers = np.where(is_upper, a ** exponents, 0.0)
                denom = 1.0 - a ** k
                scales = (1.0 - a) / denom

            return powers * scales[None, :]

        def max_window_gmean(x: np.ndarray, w: int) -> float:
            n = len(x)
            w = int(w)
            w = max(1, min(w, n))

            z = np.log(x)

            c = np.cumsum(np.r_[0.0, z])
            window_sums = c[w:] - c[:-w]
            return float(np.exp(np.max(window_sums) / w))

        def score_function(
            vec_fast: np.ndarray,  # shape (n_fast,)
            vec_slow: np.ndarray,  # shape (n_slow,)
            vec_sgnl: np.ndarray,  # shape (n_sgnl,)
            weights_fast: np.ndarray,  # shape (n_fast,)
            weights_slow: np.ndarray,  # shape (n_slow,)
            weights_sgnl: np.ndarray,  # shape (n_sgnl,)
            V: np.ndarray, 
            y: np.ndarray, 
            p: float
        ):
            n = V.shape[1]

            vec_fast = np.asarray(vec_fast, float)
            vec_slow = np.asarray(vec_slow, float)
            vec_sgnl = np.asarray(vec_sgnl, float)

            weights_fast = np.asarray(weights_fast, float)
            weights_slow = np.asarray(weights_slow, float)
            weights_sgnl = np.asarray(weights_sgnl, float)
            
            fast_res = sum(w * geom_upper_tri(a, n) for a, w in zip(vec_fast, weights_fast))
            slow_res = sum(w * geom_upper_tri(a, n) for a, w in zip(vec_slow, weights_slow))
            sgnl_res = sum(w * geom_vector(a, n)    for a, w in zip(vec_sgnl, weights_sgnl))

            res_vec = V @ ( (fast_res - slow_res) @ (e_last - sgnl_res) )
            
            perm = np.argsort(res_vec)
            y_perm = y[perm]

            wlen = int(np.round(p * len(y_perm))) if 0 < p <= 1 else int(p)
            wlen = max(1, min(wlen, len(y_perm)))

            return max_window_gmean(y_perm, wlen)

        def optimize_params_de(
                V, y, p,
                n_fast, n_slow, n_sgnl,
                maxiter=200, seed=0):
            """
            Decision vector x =
            [vec_fast (n_fast), vec_slow (n_slow), vec_sgnl (n_sgnl),
            w_fast   (n_fast), w_slow   (n_slow), w_sgnl   (n_sgnl)]
            Bounds:
            vec_* in [-1, 1], weights in [0, 1]
            """

            # bounds: a's in [-1,1], weights in [0,1]
            bounds = [(-0.95, 0.95)] * (n_fast + n_slow + n_sgnl) + [(0.0, 1.0)] * (n_fast + n_slow + n_sgnl)

            def unpack(x):
                x = np.asarray(x, float)
                i = 0
                vec_fast = x[i:i+n_fast]; i += n_fast
                vec_slow = x[i:i+n_slow]; i += n_slow
                vec_sgnl = x[i:i+n_sgnl]; i += n_sgnl
                w_fast   = x[i:i+n_fast]; i += n_fast
                w_slow   = x[i:i+n_slow]; i += n_slow
                w_sgnl   = x[i:i+n_sgnl]; i += n_sgnl
                
                eps = 1e-10
                w_fast = w_fast / (np.sum(w_fast) + eps)
                w_slow = w_slow / (np.sum(w_slow) + eps)
                w_sgnl = w_sgnl / (np.sum(w_sgnl) + eps)
    
                return vec_fast, vec_slow, vec_sgnl, w_fast, w_slow, w_sgnl

            def obj(x):
                vec_fast, vec_slow, vec_sgnl, w_fast, w_slow, w_sgnl = unpack(x)

                return -score_function(
                    vec_fast, vec_slow, vec_sgnl,
                    w_fast, w_slow, w_sgnl,
                    V, y, p
                )

            res = differential_evolution(obj, bounds=bounds, maxiter=maxiter, polish=True, seed=seed)
            xbest = res.x
            best_score = -res.fun
            return (*unpack(xbest), best_score, res)

        V = close_tr
        y = ytr_vals
        p = q_len_macd
        e_last = np.zeros(V.shape[1], float)
        e_last[-1] = 1.0
        vec_fast, vec_slow, vec_sgnl, w_fast, w_slow, w_sgnl, best_score, res = optimize_params_de(
            V, y, p, n_fast=num_fast, n_slow=num_slow, n_sgnl=num_sgnl, maxiter=300
        )
        
        # --- logging all ---
        logger.info(f"  MACDEx Optimization Results: best score: {best_score} | success: {res.success} | nit: {res.nit}")
        logger.info(f"    vec_fast: {vec_fast} | w_fast: {w_fast}")
        logger.info(f"    vec_slow: {vec_slow} | w_slow: {w_slow}")
        logger.info(f"    vec_sgnl: {vec_sgnl} | w_sgnl: {w_sgnl}")

        # --- Recompute best resulting feature vector for train/test (for slope + test feature) ---
        n = V.shape[1]
        fast = sum(w * geom_upper_tri(a, n) for a, w in zip(vec_fast, w_fast))
        slow = sum(w * geom_upper_tri(a, n) for a, w in zip(vec_slow, w_slow))
        sgnl = sum(w * geom_vector(a, n)    for a, w in zip(vec_sgnl, w_sgnl))
        R  = (fast - slow) @ (e_last - sgnl)
        
        feat_tr = V @ R
        feat_te = close_te @ R

        def best_window_masks_from_feat(feat_tr, feat_te, y, p, eps=1e-10):
            feat_tr = np.asarray(feat_tr).ravel()
            feat_te = np.asarray(feat_te).ravel()
            y = np.asarray(y).ravel()
            n = len(feat_tr)

            perm = np.argsort(feat_tr)
            y_perm = np.log(y[perm])
            feat_sorted = feat_tr[perm]

            w = int(np.round(p * n)) if 0 < p <= 1 else int(p)
            w = max(1, min(w, n))

            cs = np.cumsum(np.r_[0.0, y_perm])
            win_sums = cs[w:] - cs[:-w]
            start = int(np.argmax(win_sums))
            sl = slice(start, start + w)

            qval_lo = float(np.min(feat_sorted[sl]))
            qval_hi = float(np.max(feat_sorted[sl]))

            # quantiles of these threshold values within the *full* feat_tr distribution
            q_lo = float(np.mean(feat_tr <= qval_lo + eps))
            q_hi = float(np.mean(feat_tr <= qval_hi + eps))

            mask_tr = (feat_tr >= qval_lo) & (feat_tr <= qval_hi)
            mask_te = (feat_te >= qval_lo) & (feat_te <= qval_hi)

            return mask_tr, mask_te, qval_lo, qval_hi, q_lo, q_hi

        # ---- usage / logging ----
        mask_tr, mask_te, qval_lo, qval_hi, q_lo, q_hi = best_window_masks_from_feat(feat_tr, feat_te, y, p)

        logger.info(
            "  MACDEx -> q=[%.4f, %.4f] | qval=[%.6g, %.6g] | train kept: %.2f%% | test kept: %.2f%%",
            q_lo, q_hi, qval_lo, qval_hi,
            100 * mask_tr.mean(), 100 * mask_te.mean()
        )

        return feat_tr, feat_te, mask_tr, mask_te