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

import logging
logger = logging.getLogger(__name__)

class StratDoubleDrop(BaseStrategy):
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
        "FilterSamples_cat_volumeprice_w20_q0.7": False,
        "FilterSamples_days_to_train_end": 255 * 6,
        
        "doubledrop_thr_tr" : 0.82,
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
        
        # -- RF gate ---
        opt_params["catsample_max_lag"] = 0 #trial.suggest_int("catsample_max_lag", 0, 155, step=5) # Note: try 100 at some point
        opt_params["max_features_colinsampling"] = trial.suggest_int("max_features_colinsampling", 30, 150, step=5)
        opt_params["threshold_colin_sampling"] = trial.suggest_float("threshold_colin_sampling", 0.08, 0.11, log=True)
        
        opt_params["keep_perday_rf"]               = 3
        
        opt_params["LGB_num_boost_round"]           = 355 #trial.suggest_int("LGB_num_boost_round", 105, 455, step=50)
        opt_params["LGB_lambda_l1"]                 = 0.000784 #trial.suggest_float("LGB_lambda_l1", 0.000001, 0.001, log=True)
        opt_params["LGB_lambda_l2"]                 = 0.0 #trial.suggest_float("LGB_lambda_l2", 0.000001, 0.0001, log=True)
        opt_params["LGB_feature_fraction"]          = trial.suggest_float("LGB_feature_fraction", 0.8, 1.0)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 150, 250, step=10)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 10, 35, step=1)
        opt_params["LGB_learning_rate"]             = 0.011265 #trial.suggest_float("LGB_learning_rate", 0.0001, 0.1, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 1800, 2400, step=10)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 0.00005, 0.01, log=True)
        opt_params["LGB_path_smooth"]               = 0.210775 #trial.suggest_float("LGB_path_smooth", 0.1, 0.9, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = 0.213247 #trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        opt_params["LGB_max_bin"]                   = 200 #trial.suggest_int("LGB_max_bin", 100, 800, step=25)
        opt_params["LGB_early_stopping_rounds"]     = opt_params["LGB_num_boost_round"]//10
        
        opt_params["inc_weight_features"]   = False
        if opt_params["inc_weight_features"]:
            opt_params["weight_wndw_ratio"]     = 0.015 #trial.suggest_float("weight_wndw_ratio", 0.01, 0.5, log=True)
            opt_params["min_weight"]            = 1.654769 #trial.suggest_float("min_weight", 0.1, 2.5, step=0.1)
            opt_params["trunc_weight_features"] = 3 #trial.suggest_int("trunc_weight_features", 1, 4)
            
        # --- SLTP gate ---
        opt_params["sltp_feats_imask"] = 3 #trial.suggest_categorical("sltp_feats_imask", [0b10, 0b11, 0b01])
        opt_params["sltp_n_clusters"] = 1 #trial.suggest_int("sltp_n_clusters", 0, 2)
        
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
        
        # -- score through RF gate ---
        score_tr = np.zeros(Xtr_tree.shape[0], dtype=np.float32)
        score_te = np.zeros(Xte_tree.shape[0], dtype=np.float32)

        catsample_max_lag = opt_params["catsample_max_lag"]
        m_features = opt_params["max_features_colinsampling"]
        thr_colin = opt_params["threshold_colin_sampling"]
                
        ## GET TREENAME MASKS
        tn = np.asarray(treenames, str)
        mask_tn = np.ones(tn.shape[0], dtype=bool)
        # Categorical subsampling of treenames
        mask_tn_lag, mask_tn_noncolin = self.catsample_treenames(
            treenames,
            max_lag=catsample_max_lag,
        )
        mask_tn &= mask_tn_lag | mask_tn_noncolin
        
        # Colinear sampling based on target correlation
        mask_tn_colin_input = mask_tn_lag & ~mask_tn_noncolin
        mask_treenames_colinsampling = HelperFunctions.colinearity_treenames_mask(
            Xtr_tree[:, mask_tn_colin_input],
            ytr_opt,
            treenames=tn[mask_tn_colin_input],
            m_features=m_features,
            thr_crosscorr=thr_colin,
        )
        mask_tn[mask_tn_colin_input] = mask_treenames_colinsampling
    
        feat_tr_res = Xtr_tree[:, mask_tn]
        feat_te_res = Xte_tree[:, mask_tn]
        
        # weight features
        if opt_params["inc_weight_features"]:
            trunc_weights = opt_params.get("trunc_weight_features", 4)
            weightfeat_tr, weightfeat_te, wfeat_names = self._weight_features(Xtr_tree, Xte_tree, treenames, trunc=trunc_weights)

            ws: WeightSamples = WeightSamples(
                feat_train = weightfeat_tr, 
                y_train = ytr_opt, 
                treenames = wfeat_names,
                feat_test = weightfeat_te, 
            )
            weights = ws.establish_weights(n_bin=15, min_bd=opt_params["min_weight"], wndw_ratio=opt_params["weight_wndw_ratio"])
        else:
            weights = None
        
        #LGBModel
        mm: MachineModels = MachineModels(opt_params)
        model, _ = mm.run_LGB(feat_tr_res, ytr_opt, weights=weights)
        scores_tr_rf = model.predict(feat_tr_res, num_iteration=model.best_iteration)
        scores_te_rf = model.predict(feat_te_res, num_iteration=model.best_iteration)
        
        keep_perday = opt_params["keep_perday_rf"]
        mask_tr_rf = (
            meta_train
            .with_columns(pl.Series(scores_tr_rf).alias("score"))
            .with_columns(pl.col("score")
                .rank(method="random", descending=True)
                .over("date").alias("scores_rank")
            )
            .select(pl.col("scores_rank") <= keep_perday).to_numpy().flatten()
        )
        mask_te_rf = (
            meta_test
            .with_columns(pl.Series(scores_te_rf).alias("score"))
            .with_columns(pl.col("score")
                .rank(method="random", descending=True)
                .over("date").alias("scores_rank")
            )
            .select(pl.col("scores_rank") <= keep_perday).to_numpy().flatten()
        )
        score_tr = scores_tr_rf
        score_te = scores_te_rf
        
        mask_tr_prelgb = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_te_prelgb = np.ones(Xte_tree.shape[0], dtype=bool)
        mask_train = mask_tr_rf
        mask_test = mask_te_rf
                
        # --- SL/TP matrices ---
        imask = opt_params["sltp_feats_imask"]
        n_clusters = opt_params["sltp_n_clusters"]
        Xtr_feats, Xte_feats = self._sltp_clust_features(
            Xtr_tree,
            Xte_tree,
            treenames,
            imask=imask,
        )
        clust_sl_tr_mat, clust_sl_te_mat, clust_tp_tr_mat, clust_tp_te_mat = HelperSLTP.sltp_clustered(
            ytr_tree[mask_tr_prelgb], 
            ytr_tree_low[mask_tr_prelgb], 
            ytr_tree_high[mask_tr_prelgb], 
            ytr_tree_open[mask_tr_prelgb], 
            Xtr_feats = Xtr_feats[mask_tr_prelgb],
            Xte_feats = Xte_feats[mask_te_prelgb],
            n_clusters = n_clusters,
            include_live_mask=False, # True is doing worse
        )
        
        sl_tr = np.ones((Xtr_tree.shape[0], ytr_tree.shape[1]), dtype=float) * 0.7
        sl_te = np.ones((Xte_tree.shape[0], ytr_tree.shape[1]), dtype=float) * 0.7
        tp_tr = np.ones((Xtr_tree.shape[0], ytr_tree.shape[1]), dtype=float) * 1.8
        tp_te = np.ones((Xte_tree.shape[0], ytr_tree.shape[1]), dtype=float) * 1.8
        sl_tr[mask_tr_prelgb] = clust_sl_tr_mat
        sl_te[mask_te_prelgb] = clust_sl_te_mat
        tp_tr[mask_tr_prelgb] = clust_tp_tr_mat
        tp_te[mask_te_prelgb] = clust_tp_te_mat

        logger.info(f"  Precompute -> sl {np.median(clust_sl_tr_mat, axis=0)} | tp: {np.median(clust_sl_tr_mat, axis=0)}")

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
        cat_train, cat_test = fs.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test
            
        mask_recent_training = fs.get_recent_training_mask()
        mask_train &= mask_recent_training
        
        # --- doubledrop masks ---
        close_tr = Xtr_time[mask_train][:, :, 0].astype(np.float64, copy=False)
        close_te = Xte_time[mask_test][:, :, 0].astype(np.float64, copy=False)
        
        # --- Compute slopes via linear regression ---
        mask_tr = (close_tr[:, -1] < close_tr[:, -2]) & (close_tr[:, -2] < close_tr[:, -3])
        mask_te = (close_te[:, -1] < close_te[:, -2]) & (close_te[:, -2] < close_te[:, -3])
        
        feat_tr = close_tr[:, -1] / close_tr[:, -3]
        feat_te = close_te[:, -1] / close_te[:, -3]
        
        qthr = self.precompute_params.get("doubledrop_thr_tr")
        mask_tr &= feat_tr > qthr
        mask_te &= feat_te > qthr
        
        logger.info(f"  DoubleDrop -> quantile thr value: {qthr}")
            
        # -- log info ---
        mask_train[mask_train] = mask_tr
        mask_test[mask_test] = mask_te
        
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
    def _sltp_clust_features(
        self,
        Xtr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        treenames: list[str],
        imask: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Selects a subset of tree features using a binary integer mask.

        The integer `imask` is interpreted as a binary mask over a fixed,
        ordered list of feature names. A bit value of 1 means the feature
        is selected; 0 means it is ignored. The most-significant bit maps
        to the first feature in the list.

        Examples
        --------
        >>> feats = [
        ...     "MathFeature_Return_log",
        ...     "FeatureGroup_WeightedIndexPct",
        ...     "MathFeature_PriceAdjustment",
        ...     "FeatureGroup_WeightedIndexMHPct_lag_m1_MH_2",
        ...     "FeatureGroup_AvgReturnPct_lag_m2",
        ... ]
        >>> imask = 4
        >>> bin(imask)
        '0b10'
        >>> # Mask (padded to len(feats)):
        >>> # [False, False, False, True, False]
        >>> # Selected feature:
        >>> # "FeatureGroup_WeightedIndexMHPct_lag_m1_MH_2"
        """
        feats = [
            "MathFeature_Return_log",
            "FeatureGroup_WeightedIndexMHPct_lag_m1_MH_2",
        ]

        mask = ((imask >> np.arange(len(feats))) & 1).astype(bool)[::-1]

        feats_arr = np.array(feats)
        selected_feats = feats_arr[mask].tolist()

        idx = [treenames.index(f) for f in selected_feats]

        return Xtr_tree[:, idx], Xte_tree[:, idx]
    
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
                
    def _rf_gate_sample_masks(
        self,
        feat_tr: np.ndarray,          # shape (n_samples, n_features)
        feat_te: np.ndarray,          # shape (n_samples, n_features)
        y_gate_tr: np.ndarray,          # ytr_tree[:, -1]
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train a RF regressor on 3 gating features (ATR, slope, RMSE) -> y_gate_tr,
        then select samples by thresholds on:
          - predicted value range (quantiles)
          - optional uncertainty proxy (tree prediction std)
          - optional train residual (abs error) cap

        Returns:
          mask_train, mask_test
        """
        rf_gate = RandomForestRegressor(
            n_estimators=opt_params.get("rf_gate_n_estimators", 400),
            max_depth=opt_params.get("rf_gate_max_depth", None),
            min_samples_split=opt_params.get("rf_gate_min_samples_split", 2),
            min_samples_leaf=opt_params.get("rf_gate_min_samples_leaf", 1),
            max_features=opt_params.get("rf_gate_max_features", "sqrt"),
            bootstrap=opt_params.get("rf_gate_bootstrap", True),
            oob_score=opt_params.get("rf_gate_oob_score", False) if opt_params.get("rf_gate_bootstrap", True) else False,
            n_jobs=opt_params.get("rf_gate_n_jobs", -1),
            max_samples=0.2 if opt_params.get("rf_gate_bootstrap", True) else None,
        )
        rf_gate.fit(feat_tr, y_gate_tr)

        scores_tr = rf_gate.predict(feat_tr)
        scores_te = rf_gate.predict(feat_te)

        return scores_tr, scores_te
    
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
    
    def _strech_ohlc(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Streches OHLC arrays to [0,1] range per sample.
        """
        center = 0.5
        D = np.abs(close_arr - center)                       # (N, T)
        den = np.max(D, axis=1)                              # (N,)
        den = np.where(den < 1e-5, 1.0, den)                 # avoid div-by-zero

        close_strch = ((close_arr - center) / den[:, np.newaxis])/2 + center
        open_strch  = ((open_arr - center) / den[:, np.newaxis])/2 + center
        high_strch  = ((high_arr - center) / den[:, np.newaxis])/2 + center
        low_strch   = ((low_arr - center) / den[:, np.newaxis])/2 + center

        return open_strch, high_strch, low_strch, close_strch