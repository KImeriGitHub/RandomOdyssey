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

import logging
logger = logging.getLogger(__name__)

class StratSLTPQuantileWindowCascade(BaseStrategy):
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
        "dd_active": True,
        "td_active": False,
        "atr_active": False,
        "nj_active": False,
        "tun_active": False,
        "macd_active": False,
        "rf_active": True,
        
        "all_deactivated": False,
        "q_len_dd": 0.265,
        "weight_dd_ma": 0.4,
        "q_len_dd_ma": 0.95,
    }

    base_params = {
    }

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {
            "rf_active": True,
            
            "catsample_max_lag": 0,
            "max_features_colinsampling": 30,
            "threshold_colin_sampling": 0.080767,
            
            "LGB_num_boost_round": 435,
            "LGB_lambda_l1": 0.005,
            "LGB_lambda_l2": 0.00004,
            "LGB_feature_fraction": 1.0,
            "LGB_num_leaves": 250,
            "LGB_max_depth": 80,
            "LGB_learning_rate": 0.01,
            "LGB_min_data_in_leaf": 700,
            "LGB_min_gain_to_split": 0.000001,
            "LGB_path_smooth": 0.6,
            "LGB_min_sum_hessian_in_leaf": 0.2,
            "LGB_max_bin": 250,
            "LGB_early_stopping_rounds": 43,
        }
        
        # --- weight params ---
        opt_params["weight_wndw_ratio"] = trial.suggest_float("weight_wndw_ratio", 0.01, 0.1, log=True)
        opt_params["min_weight"] = trial.suggest_float("min_weight", 0.1, 2.5, log=True)
        opt_params["trunc_weight_features"] = trial.suggest_int("trunc_weight_features", 1, 4)
            
        # --- SLTP gate ---
        opt_params["sltp_feats_imask"] = trial.suggest_categorical("sltp_feats_imask", [0b100, 0b001, 0b110, 0b111])
        opt_params["sltp_n_clusters"] = trial.suggest_int("sltp_n_clusters", 2, 20, step=2)
        
        opt_params["keep_perday_rf"] = trial.suggest_int("keep_perday_rf", 3, 7)
        
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
        mask_tr = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_te = np.ones(Xte_tree.shape[0], dtype=bool)
        score_tr = np.zeros(Xtr_tree.shape[0], dtype=np.float32)
        score_te = np.zeros(Xte_tree.shape[0], dtype=np.float32)
        if opt_params["rf_active"]:
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
            trunc_weights = opt_params.get("trunc_weight_features", 4)
            weightfeat_tr, weightfeat_te, wfeat_names = self._weight_features(Xtr_tree, Xte_tree, treenames, trunc=trunc_weights)

            ws: WeightSamples = WeightSamples(
                feat_train = weightfeat_tr, 
                y_train = ytr_opt, 
                treenames = wfeat_names,
                feat_test = weightfeat_te, 
            )
            weights = ws.establish_weights(n_bin=15, min_bd=opt_params["min_weight"], wndw_ratio=opt_params["weight_wndw_ratio"])
            
            #LGBModel
            mm: MachineModels = MachineModels(opt_params)
            model, _ = mm.run_LGB(feat_tr_res, ytr_opt, weights=weights)
            scores_tr_rf = model.predict(feat_tr_res, num_iteration=model.best_iteration)
            scores_te_rf = model.predict(feat_te_res, num_iteration=model.best_iteration)
            
            keep_perday = opt_params["keep_perday_rf"]
            mask_tr_rf = (
                meta_train
                .filter(mask_tr)
                .with_columns(pl.Series(scores_tr_rf).alias("score"))
                .with_columns(pl.col("score")
                    .rank(method="random", descending=True)
                    .over("date").alias("scores_rank")
                )
                .select(pl.col("scores_rank") <= keep_perday).to_numpy().flatten()
            )
            mask_te_rf = (
                meta_test
                .filter(mask_te)
                .with_columns(pl.Series(scores_te_rf).alias("score"))
                .with_columns(pl.col("score")
                    .rank(method="random", descending=True)
                    .over("date").alias("scores_rank")
                )
                .select(pl.col("scores_rank") <= keep_perday).to_numpy().flatten()
            )
            score_tr[mask_tr] = scores_tr_rf
            score_te[mask_te] = scores_te_rf
            
            mask_tr_prelgb = mask_tr.copy()
            mask_te_prelgb = mask_te.copy()
            mask_tr[mask_tr] = mask_tr_rf
            mask_te[mask_te] = mask_te_rf
                
        # --- SL/TP matrices ---
        imask = opt_params.get("sltp_feats_imask", 0b100)
        n_clusters = opt_params.get("sltp_n_clusters", 1000)
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

        return mask_tr, mask_te, sl_tr, sl_te, tp_tr, tp_te, score_tr, score_te

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
        
        mask_tr = np.zeros(Xtr_tree.shape[0], dtype=bool)
        mask_te = np.zeros(Xte_tree.shape[0], dtype=bool)
        
        # --- doubledrop masks ---
        dates_digitized = (
            meta_train.select(
                (pl.col("date").to_physical() - pl.col("date").to_physical().min())
            )
            .to_numpy().flatten()
        )
        dates_digitized = dates_digitized / dates_digitized.max()
        if self.precompute_params["dd_active"]:
            feat_tr_dd, feat_te_dd, mask_tr_dd, mask_te_dd = self._compute_doubledrop_feat(
                Xtr_time,
                Xte_time,
                ytr_tree[:, -1],
                dates_digitized,
                self.precompute_params,
            )
            mask_tr = mask_tr | mask_tr_dd
            mask_te = mask_te | mask_te_dd

        # -- log info ---
        logger.info(
            "  QWC -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_tr.mean(), 100 * mask_te.mean() 
        )
        n_dates_tr = meta_train.get_column('date').n_unique()
        n_dates_te = meta_test.get_column('date').n_unique()
        
        n_dates_masked_tr = meta_train.filter(pl.Series(mask_tr)).get_column('date').n_unique()
        n_dates_masked_te = meta_test.filter(pl.Series(mask_te)).get_column('date').n_unique()
        
        logger.info(f"  Dates kept -> train: {n_dates_masked_tr / n_dates_tr} | test: {n_dates_masked_te / n_dates_te}")


        Xtr_feats, Xte_feats = self._sltp_clust_features(
            Xtr_tree,
            Xte_tree,
            treenames,
            imask=0b100,
        )
        clust_sl_tr_mat, clust_sl_te_mast, clust_tp_tr_mat, clust_tp_te_mat = HelperSLTP.sltp_clustered(
            ytr_tree, 
            ytr_tree_low, 
            ytr_tree_high, 
            ytr_tree_open, 
            Xtr_feats = Xtr_feats,
            Xte_feats = Xte_feats,
            n_clusters = 100,
            include_live_mask=False,
        )
        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_tr.mean(),
            100 * mask_te.mean(),
        )
        logger.info(f"  Precompute -> sl {np.median(clust_sl_tr_mat, axis=0)} | tp: {np.median(clust_sl_tr_mat, axis=0)}")

        return mask_tr, mask_te, clust_sl_tr_mat, clust_sl_te_mast, clust_tp_tr_mat, clust_tp_te_mat
    
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
            "MathFeature_PriceAdjustment",
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
            "MathFeature_PriceAdjustment",
            "FeatureGroup_AvgReturnPct_lag_m2",
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
    
    def _compute_atr_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        ytr_vals: np.ndarray,      # (n_samples,)
        dates_digitized: np.ndarray,          # (n_samples,)
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        # --- parameters ---
        q_len_atr = opt_params["q_len_atr"]
        q_len_atr_slope = opt_params["q_len_atr_slope"]
        weight_atr = opt_params.get("weight_atr", 0.8)
        weight_atr_slope = opt_params.get("weight_atr_slope", 0.8)

        # --- Extract close / high / low ---
        open_tr, high_tr, low_tr, close_tr = self._get_ohlc_from_Xtime(Xtr_time)
        open_te, high_te, low_te, close_te = self._get_ohlc_from_Xtime(Xte_time)
        
        # streching
        _, high_tr_strch, low_tr_strch, close_tr_strch = self._strech_ohlc(
            open_tr, high_tr, low_tr, close_tr
        )
        _, high_te_strch, low_te_strch, close_te_strch = self._strech_ohlc(
            open_te, high_te, low_te, close_te
        )

        # --- True Range and ATR (EMA of TR) and Slope ---
        range_tr = np.abs(high_tr_strch[:, 1:] - low_tr_strch[:, 1:])
        jump_to_high_tr = np.abs(high_tr_strch[:, 1:] - close_tr_strch[:, :-1])
        jump_to_low_tr = np.abs(low_tr_strch[:, 1:] - close_tr_strch[:, :-1])
        true_range_tr = np.maximum.reduce([range_tr, jump_to_high_tr, jump_to_low_tr])
        
        range_te = np.abs(high_te_strch[:, 1:] - low_te_strch[:, 1:])
        jump_to_high_te = np.abs(high_te_strch[:, 1:] - close_te_strch[:, :-1])
        jump_to_low_te = np.abs(low_te_strch[:, 1:] - close_te_strch[:, :-1])
        true_range_te = np.maximum.reduce([range_te, jump_to_high_te, jump_to_low_te])
        
        diff_time_tr = np.diff(close_tr_strch, axis=1)  # shape (nS, nT-1)
        diff_time_te = np.diff(close_te_strch, axis=1)  # shape (nS, nT-1)
        
        # --- Features for different windows in one matrix ---
        atr_wndws = np.arange(4, 15)
        atr_tr_front = np.column_stack([
            HelperFunctions.moving_average(true_range_tr, wndw)[:,-1]
            for wndw in atr_wndws
        ])  # shape (n_samples, n_wndws)
        
        slope_wnds = np.arange(6, 15)
        slope_tr_front = np.column_stack([
            np.mean(diff_time_tr[:, -(wndw+2):], axis=1)
            for wndw in slope_wnds
        ])  # shape (n_samples, n_wndws)
                
        # --- Maximize through quantile windows ---
        ymean_atr_vals, qlow_val_atr, qhigh_val_atr, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=atr_tr_front,
            y=ytr_vals,
            q_len=q_len_atr,
            dates_digitized=dates_digitized,
            weight_mean=weight_atr,
            feat_name="ATR",
        )
        
        best_atr_idx = int(np.nanargmax(ymean_atr_vals))
        wndw_atr_best = int(atr_wndws[best_atr_idx])

        # --- Build atr features and masks ---
        feat_atr_tr = atr_tr_front[:, best_atr_idx]
        feat_atr_te = HelperFunctions.moving_average(true_range_te, wndw_atr_best)[:,-1]

        mask_tr_atr = (feat_atr_tr >= qlow_val_atr[best_atr_idx]) & (feat_atr_tr <= qhigh_val_atr[best_atr_idx])
        mask_te_atr = (feat_atr_te >= qlow_val_atr[best_atr_idx]) & (feat_atr_te <= qhigh_val_atr[best_atr_idx])
        
        # --- Slope maximization ---
        ymean_slope_vals, qlow_val_slope, qhigh_val_slope, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=slope_tr_front[mask_tr_atr],
            y=ytr_vals[mask_tr_atr],
            q_len=q_len_atr_slope,
            dates_digitized=dates_digitized[mask_tr_atr],
            weight_mean=weight_atr_slope,
            feat_name="ATR_Slope",
        )
        best_slope_idx = int(np.nanargmax(ymean_slope_vals))
        wndw_slope_best = int(slope_wnds[best_slope_idx])
        
        # --- Build slope features and masks ---
        feat_slope_tr = slope_tr_front[:, best_slope_idx]
        feat_slope_te = np.mean(diff_time_te[:, -(wndw_slope_best+2):], axis=1)
        
        mask_tr_slope = (feat_slope_tr >= qlow_val_slope[best_slope_idx]) & (feat_slope_tr <= qhigh_val_slope[best_slope_idx])
        mask_te_slope = (feat_slope_te >= qlow_val_slope[best_slope_idx]) & (feat_slope_te <= qhigh_val_slope[best_slope_idx])
        
        mask_tr_slope[~mask_tr_atr] = False
        mask_te_slope[~mask_te_atr] = False
        
        # --- Combine ---
        mask_tr = mask_tr_atr & mask_tr_slope
        mask_te = mask_te_atr & mask_te_slope
        
        feats_tr = np.column_stack([feat_atr_tr, feat_slope_tr])
        feats_te = np.column_stack([feat_atr_te, feat_slope_te])
        
        # --- Logging ---
        logger.info(
            "  ATR -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_tr.mean(), 100 * mask_te.mean() 
        )

        return feats_tr, feats_te, mask_tr, mask_te
        
    def _compute_doubledrop_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        ytr_vals: np.ndarray,      # (n_samples,)
        dates_digitized: np.ndarray,          # (n_samples,)
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute masks based on slope of of price series in a certain window.
        """
        
        # --- Parameters ---
        q_len_dd_ma = opt_params["q_len_dd_ma"]
        weight_dd_ma = opt_params.get("weight_dd_ma", 0.8)
        
        # --- Close prices ---
        close_tr = Xtr_time[:, :, 0].astype(np.float64, copy=False)
        close_te = Xte_time[:, :, 0].astype(np.float64, copy=False)
        
        # --- Compute slopes via linear regression ---
        mask_tr_drop = (close_tr[:, -1] < close_tr[:, -2]) & (close_tr[:, -2] < close_tr[:, -3])
        mask_te_drop = (close_te[:, -1] < close_te[:, -2]) & (close_te[:, -2] < close_te[:, -3])
        
        # --- Compute MA features ---
        ma_wndws = np.arange(1, 15)
        ma_tr_front = np.column_stack([
            HelperFunctions.moving_average(close_tr, wndw)[:,-1]
            for wndw in ma_wndws
        ])  # shape (n_samples, n_wndws)
        
        # --- Maximize through quantile windows ---
        ymean_ma_vals, qlow_val_ma, qhigh_val_ma, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=ma_tr_front[mask_tr_drop],
            y=ytr_vals[mask_tr_drop],
            q_len=q_len_dd_ma,
            dates_digitized=dates_digitized[mask_tr_drop],
            weight_mean=weight_dd_ma,
            feat_name="DD_MA",
        )
        best_ma_idx = int(np.nanargmax(ymean_ma_vals))
        wndw_ma_best = int(ma_wndws[best_ma_idx])
        
        # --- Build ma features and masks ---
        feat_ma_tr = ma_tr_front[:, best_ma_idx]
        feat_ma_te = HelperFunctions.moving_average(close_te, wndw_ma_best)[:,-1]
        
        mask_tr_ma = (feat_ma_tr >= qlow_val_ma[best_ma_idx]) & (feat_ma_tr <= qhigh_val_ma[best_ma_idx])
        mask_te_ma = (feat_ma_te >= qlow_val_ma[best_ma_idx]) & (feat_ma_te <= qhigh_val_ma[best_ma_idx])
        
        mask_tr_ma[~mask_tr_drop] = False
        mask_te_ma[~mask_te_drop] = False
        
        # --- Combine ---
        mask_tr = mask_tr_drop & mask_tr_ma
        mask_te = mask_te_drop & mask_te_ma
        
        feats_tr = feat_ma_tr
        feats_te = feat_ma_te
        
        # --- Logging ---
        logger.info(
            "  DoubleDrop -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_tr.mean(), 100 * mask_te.mean() 
        )
        
        return feats_tr, feats_te, mask_tr, mask_te
    
    def _compute_nightjump_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        ytr_vals: np.ndarray,      # (n_samples,)
        dates_digitized: np.ndarray,          # (n_samples,)
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        # --- parameters ---
        q_len_nj = opt_params["q_len_nj"]
        q_len_nj_ma = opt_params["q_len_nj_ma"]
        weight_nj = opt_params.get("weight_nj", 0.8)
        weight_nj_ma = opt_params.get("weight_nj_ma", 0.8)
        
        # --- Extract close / high / low ---
        open_tr, high_tr, low_tr, close_tr = self._get_ohlc_from_Xtime(Xtr_time)
        open_te, high_te, low_te, close_te = self._get_ohlc_from_Xtime(Xte_time)
        
        # streching
        open_tr_strch, _, _, close_tr_strch = self._strech_ohlc(
            open_tr, high_tr, low_tr, close_tr
        )
        open_te_strch, _, _, close_te_strch = self._strech_ohlc(
            open_te, high_te, low_te, close_te
        )

        # --- Night Jump ---
        night_jump_tr = open_tr_strch[:, 1:] - close_tr_strch[:, :-1]
        night_jump_te = open_te_strch[:, 1:] - close_te_strch[:, :-1]
        
        # --- Features for different windows in one matrix ---
        njma_wndws = np.arange(4, 13)
        nj_tr_front = np.column_stack([
            HelperFunctions.moving_average(night_jump_tr, wndw)[:,-1]
            for wndw in njma_wndws
        ])  # shape (n_samples, n_wndws)
                
        # --- Maximize through quantile windows ---
        ymean_nj_vals, qlow_val_nj, qhigh_val_nj, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=nj_tr_front,
            y=ytr_vals,
            q_len=q_len_nj,
            dates_digitized=dates_digitized,
            weight_mean=weight_nj,
            feat_name="NJ",
        )
        
        best_nj_idx = int(np.nanargmax(ymean_nj_vals))
        wndw_nj_best = int(njma_wndws[best_nj_idx])

        # --- Build atr features and masks ---
        feat_nj_tr = nj_tr_front[:, best_nj_idx]
        feat_nj_te = HelperFunctions.moving_average(night_jump_te, wndw_nj_best)[:,-1]

        mask_tr_nj = (feat_nj_tr >= qlow_val_nj[best_nj_idx]) & (feat_nj_tr <= qhigh_val_nj[best_nj_idx])
        mask_te_nj = (feat_nj_te >= qlow_val_nj[best_nj_idx]) & (feat_nj_te <= qhigh_val_nj[best_nj_idx])
        
        # --- Compute MA features ---
        ma_wndws = np.arange(1, 5)
        ma_tr_front = np.column_stack([
            HelperFunctions.moving_average(close_tr_strch, wndw)[:,-1]
            for wndw in ma_wndws
        ])  # shape (n_samples, n_wndws)
        
        # --- Maximize through quantile windows ---
        ymean_ma_vals, qlow_val_ma, qhigh_val_ma, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=ma_tr_front[mask_tr_nj],
            y=ytr_vals[mask_tr_nj],
            q_len=q_len_nj_ma,
            dates_digitized=dates_digitized[mask_tr_nj],
            weight_mean=weight_nj_ma,
            feat_name="NJ_MA",
        )
        best_ma_idx = int(np.nanargmax(ymean_ma_vals))
        wndw_ma_best = int(ma_wndws[best_ma_idx])
        
        # --- Build ma features and masks ---
        feat_ma_tr = ma_tr_front[:, best_ma_idx]
        feat_ma_te = HelperFunctions.moving_average(close_te_strch, wndw_ma_best)[:,-1]
        
        mask_tr_ma = (feat_ma_tr >= qlow_val_ma[best_ma_idx]) & (feat_ma_tr <= qhigh_val_ma[best_ma_idx])
        mask_te_ma = (feat_ma_te >= qlow_val_ma[best_ma_idx]) & (feat_ma_te <= qhigh_val_ma[best_ma_idx])
        
        mask_tr_ma[~mask_tr_nj] = False
        mask_te_ma[~mask_te_nj] = False
        
        # --- Combine ---
        mask_tr = mask_tr_nj & mask_tr_ma
        mask_te = mask_te_nj & mask_te_ma
        
        feats_tr = np.column_stack([feat_nj_tr, feat_ma_tr])
        feats_te = np.column_stack([feat_nj_te, feat_ma_te])
        
        # --- Logging ---
        logger.info(
            "  NightJump -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_tr.mean(), 100 * mask_te.mean() 
        )

        return feats_tr, feats_te, mask_tr, mask_te

    def _compute_tunneling_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        ytr_vals: np.ndarray,      # (n_samples,)
        dates_digitized: np.ndarray,          # (n_samples,)
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        # --- parameters ---
        q_len_tun = opt_params["q_len_tun"]
        q_len_tun_ma = opt_params["q_len_tun_ma"]
        weight_tun = opt_params.get("weight_tun", 0.8)
        weight_tun_ma = opt_params.get("weight_tun_ma", 0.8)
        
        # --- Extract close / high / low ---
        open_tr, high_tr, low_tr, close_tr = self._get_ohlc_from_Xtime(Xtr_time)
        open_te, high_te, low_te, close_te = self._get_ohlc_from_Xtime(Xte_time)
        
        # streching
        _, _, _, close_tr_strch = self._strech_ohlc(
            open_tr, high_tr, low_tr, close_tr
        )
        _, _, _, close_te_strch = self._strech_ohlc(
            open_te, high_te, low_te, close_te
        )

        # --- Tunneling MAs---
        ma_wndw = 10
        close_ma_tr = HelperFunctions.moving_average(close_tr_strch, ma_wndw)
        close_ma_te = HelperFunctions.moving_average(close_te_strch, ma_wndw)
        
        # --- Features for different windows in one matrix ---
        tun_wndws = np.arange(5, 21, 2)
        tun_tr_front = np.column_stack([
            np.sqrt(np.mean((close_tr_strch[:, -wndw:] - close_ma_tr[:, -wndw:]) ** 2, axis=1))
            for wndw in tun_wndws
        ])  # shape (n_samples, n_wndws)
        
        # --- Maximize through quantile windows ---
        ymean_tun_vals, qlow_val_tun, qhigh_val_tun, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=tun_tr_front,
            y=ytr_vals,
            q_len=q_len_tun,
            dates_digitized=dates_digitized,
            weight_mean=weight_tun,
            feat_name="TUN",
        )
        
        best_tun_idx = int(np.nanargmax(ymean_tun_vals))
        wndw_tun_best = int(tun_wndws[best_tun_idx])

        # --- Build atr features and masks ---
        feat_tun_tr = tun_tr_front[:, best_tun_idx]
        feat_tun_te = np.sqrt(np.mean((close_te_strch[:, -wndw_tun_best:] - close_ma_te[:, -wndw_tun_best:]) ** 2, axis=1))

        mask_tr_tun = (feat_tun_tr >= qlow_val_tun[best_tun_idx]) & (feat_tun_tr <= qhigh_val_tun[best_tun_idx])
        mask_te_tun = (feat_tun_te >= qlow_val_tun[best_tun_idx]) & (feat_tun_te <= qhigh_val_tun[best_tun_idx])
        
        # --- Compute MA features ---
        tun_ma_wndws = np.arange(4, 16, 2)
        tun_ma_tr_front = np.column_stack([
            HelperFunctions.moving_average(close_tr_strch, wndw)[:,-1]
            for wndw in tun_ma_wndws
        ])  # shape (n_samples, n_wndws)
        
        # --- Maximize through quantile windows ---
        ymean_tun_ma_vals, qlow_val_tun_ma, qhigh_val_tun_ma, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=tun_ma_tr_front[mask_tr_tun],
            y=ytr_vals[mask_tr_tun],
            q_len=q_len_tun_ma,
            dates_digitized=dates_digitized[mask_tr_tun],
            weight_mean=weight_tun_ma,
            feat_name="TUN_MA",
        )
        best_tun_ma_idx = int(np.nanargmax(ymean_tun_ma_vals))
        wndw_tun_ma_best = int(tun_ma_wndws[best_tun_ma_idx])
        
        # --- Build ma features and masks ---
        feat_tun_ma_tr = tun_ma_tr_front[:, best_tun_ma_idx]
        feat_tun_ma_te = HelperFunctions.moving_average(close_te_strch, wndw_tun_ma_best)[:,-1]
        
        mask_tr_tun_ma = (feat_tun_ma_tr >= qlow_val_tun_ma[best_tun_ma_idx]) & (feat_tun_ma_tr <= qhigh_val_tun_ma[best_tun_ma_idx])
        mask_te_tun_ma = (feat_tun_ma_te >= qlow_val_tun_ma[best_tun_ma_idx]) & (feat_tun_ma_te <= qhigh_val_tun_ma[best_tun_ma_idx])
        
        mask_tr_tun_ma[~mask_tr_tun] = False
        mask_te_tun_ma[~mask_te_tun] = False
        
        # --- Combine ---
        mask_tr = mask_tr_tun & mask_tr_tun_ma
        mask_te = mask_te_tun & mask_te_tun_ma
        
        feats_tr = np.column_stack([feat_tun_tr, feat_tun_ma_tr])
        feats_te = np.column_stack([feat_tun_te, feat_tun_ma_te])
        
        # --- Logging ---
        logger.info(
            "  Tunneling -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_tr.mean(), 100 * mask_te.mean() 
        )

        return feats_tr, feats_te, mask_tr, mask_te

    def _compute_macd_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        ytr_vals: np.ndarray,      # (n_samples,)
        dates_digitized: np.ndarray,  # (n_samples,)
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ### CHATGPT Generated Code - MACD Feature Computation ###
        # --- parameters ---
        q_len_macd = opt_params["q_len_macd"]
        q_len_macd_slope = opt_params["q_len_macd_slope"]
        weight_macd = opt_params.get("weight_macd", 0.8)
        weight_macd_slope = opt_params.get("weight_macd_slope", 0.8)

        # MACD search space (window and alpha are independent)
        fast_wndws = np.asarray(opt_params.get("macd_fast_wndws", np.array([8, 12])), dtype=int)
        slow_wndws = np.asarray(opt_params.get("macd_slow_wndws", np.array([12, 19])), dtype=int)

        fast_alphas = np.asarray(opt_params.get("macd_fast_alphas", np.array([0.2, 0.3])), dtype=float)
        slow_alphas = np.asarray(opt_params.get("macd_slow_alphas", np.array([0.05, 0.1])), dtype=float)

        # slope windows (mean of last diffs)
        slope_wnds = np.asarray(opt_params.get("macd_slope_wndws", np.arange(6, 15, 2)), dtype=int)

        # --- Extract and stretch close ---
        _, _, _, close_tr = self._get_ohlc_from_Xtime(Xtr_time)
        _, _, _, close_te = self._get_ohlc_from_Xtime(Xte_time)

        #_, _, _, close_tr_strch = self._strech_ohlc(open_tr, high_tr, low_tr, close_tr)
        #_, _, _, close_te_strch = self._strech_ohlc(open_te, high_te, low_te, close_te)

        # --- Build MACD candidates on train (only last value per config) ---
        configs: list[tuple[int, float, int, float]] = []
        macd_tr_cols: list[np.ndarray] = []

        for fw in fast_wndws:
            for sw in slow_wndws:
                for fa in fast_alphas:
                    if not (0.0 < fa <= 1.0):
                        continue
                    for sa in slow_alphas:
                        if not (0.0 < sa <= 1.0):
                            continue

                        configs.append((int(fw), float(fa), int(sw), float(sa)))

                        ema_fast_tr = HelperFunctions.exponential_moving_average(
                            close_tr, window=int(fw), alpha=float(fa)
                        )
                        ema_slow_tr = HelperFunctions.exponential_moving_average(
                            close_tr, window=int(sw), alpha=float(sa)
                        )
                        macd_tr = ema_fast_tr - ema_slow_tr
                        macd_tr_cols.append(macd_tr[:, -1])

        macd_tr_front = np.column_stack(macd_tr_cols)  # (n_samples, n_configs)

        # --- Maximize through quantile windows (MACD level) ---
        ymean_macd_vals, qlow_val_macd, qhigh_val_macd, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=macd_tr_front,
            y=ytr_vals,
            q_len=q_len_macd,
            dates_digitized=dates_digitized,
            weight_mean=weight_macd,
            feat_name="MACD",
        )

        best_macd_idx = int(np.nanargmax(ymean_macd_vals))
        fw_best, fa_best, sw_best, sa_best = configs[best_macd_idx]

        # --- Recompute best MACD series for train/test (for slope + test feature) ---
        ema_fast_tr = HelperFunctions.exponential_moving_average(close_tr, window=fw_best, alpha=fa_best)
        ema_slow_tr = HelperFunctions.exponential_moving_average(close_tr, window=sw_best, alpha=sa_best)
        macd_series_tr = ema_fast_tr - ema_slow_tr

        ema_fast_te = HelperFunctions.exponential_moving_average(close_te, window=fw_best, alpha=fa_best)
        ema_slow_te = HelperFunctions.exponential_moving_average(close_te, window=sw_best, alpha=sa_best)
        macd_series_te = ema_fast_te - ema_slow_te

        feat_macd_tr = macd_series_tr[:, -1]
        feat_macd_te = macd_series_te[:, -1]

        mask_tr_macd = (feat_macd_tr >= qlow_val_macd[best_macd_idx]) & (feat_macd_tr <= qhigh_val_macd[best_macd_idx])
        mask_te_macd = (feat_macd_te >= qlow_val_macd[best_macd_idx]) & (feat_macd_te <= qhigh_val_macd[best_macd_idx])

        # --- Slope maximization (conditioned on MACD mask) ---
        diff_macd_tr = np.diff(macd_series_tr, axis=1)  # (nS, nT-1)
        slope_tr_front = np.column_stack([
            np.mean(diff_macd_tr[:, -(wndw + 2):], axis=1)
            for wndw in slope_wnds
        ])  # (n_samples, n_slope_wndws)

        ymean_slope_vals, qlow_val_slope, qhigh_val_slope, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=slope_tr_front[mask_tr_macd],
            y=ytr_vals[mask_tr_macd],
            q_len=q_len_macd_slope,
            dates_digitized=dates_digitized[mask_tr_macd],
            weight_mean=weight_macd_slope,
            feat_name="MACD_Slope",
        )

        best_slope_idx = int(np.nanargmax(ymean_slope_vals))
        wndw_slope_best = int(slope_wnds[best_slope_idx])

        feat_slope_tr = slope_tr_front[:, best_slope_idx]

        diff_macd_te = np.diff(macd_series_te, axis=1)
        feat_slope_te = np.mean(diff_macd_te[:, -(wndw_slope_best + 2):], axis=1)

        mask_tr_slope = (feat_slope_tr >= qlow_val_slope[best_slope_idx]) & (feat_slope_tr <= qhigh_val_slope[best_slope_idx])
        mask_te_slope = (feat_slope_te >= qlow_val_slope[best_slope_idx]) & (feat_slope_te <= qhigh_val_slope[best_slope_idx])

        mask_tr_slope[~mask_tr_macd] = False
        mask_te_slope[~mask_te_macd] = False

        # --- Combine ---
        mask_tr = mask_tr_macd & mask_tr_slope
        mask_te = mask_te_macd & mask_te_slope

        feats_tr = np.column_stack([feat_macd_tr, feat_slope_tr])
        feats_te = np.column_stack([feat_macd_te, feat_slope_te])

        logger.info(
            "  MACD(fw=%d,fa=%.3f, sw=%d,sa=%.3f) -> train kept: %.2f%% | test kept: %.2f%%",
            fw_best, fa_best, sw_best, sa_best,
            100 * mask_tr.mean(), 100 * mask_te.mean()
        )

        return feats_tr, feats_te, mask_tr, mask_te

    def _compute_tripledrop_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        ytr_vals: np.ndarray,      # (n_samples,)
        dates_digitized: np.ndarray,          # (n_samples,)
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute masks based on slope of of price series in a certain window.
        """
        
        # --- Parameters ---
        q_len_td_ma = opt_params["q_len_td_ma"]
        weight_td_ma = opt_params.get("weight_td_ma", 0.8)
        
        # --- Close prices ---
        close_tr = Xtr_time[:, :, 0].astype(np.float64, copy=False)
        close_te = Xte_time[:, :, 0].astype(np.float64, copy=False)
        
        # --- Compute slopes via linear regression ---
        mask_tr_drop = (close_tr[:, -1] < close_tr[:, -2]) & (close_tr[:, -2] < close_tr[:, -3]) & (close_tr[:, -3] < close_tr[:, -4])
        mask_te_drop = (close_te[:, -1] < close_te[:, -2]) & (close_te[:, -2] < close_te[:, -3]) & (close_te[:, -3] < close_te[:, -4])
        
        # --- Compute MA features ---
        ma_wndws = np.arange(1, 14)
        ma_tr_front = np.column_stack([
            HelperFunctions.moving_average(close_tr, wndw)[:,-1]
            for wndw in ma_wndws
        ])  # shape (n_samples, n_wndws)
        
        # --- Maximize through quantile windows ---
        ymean_ma_vals, qlow_val_ma, qhigh_val_ma, _, _ = HelperFunctions.maximize_through_quantile_windows(
            matrix=ma_tr_front[mask_tr_drop],
            y=ytr_vals[mask_tr_drop],
            q_len=q_len_td_ma,
            dates_digitized=dates_digitized[mask_tr_drop],
            weight_mean=weight_td_ma,
            feat_name="TD_MA",
        )
        best_ma_idx = int(np.nanargmax(ymean_ma_vals))
        wndw_ma_best = int(ma_wndws[best_ma_idx])
        
        # --- Build ma features and masks ---
        feat_ma_tr = ma_tr_front[:, best_ma_idx]
        feat_ma_te = HelperFunctions.moving_average(close_te, wndw_ma_best)[:,-1]
        
        mask_tr_ma = (feat_ma_tr >= qlow_val_ma[best_ma_idx]) & (feat_ma_tr <= qhigh_val_ma[best_ma_idx])
        mask_te_ma = (feat_ma_te >= qlow_val_ma[best_ma_idx]) & (feat_ma_te <= qhigh_val_ma[best_ma_idx])
        
        mask_tr_ma[~mask_tr_drop] = False
        mask_te_ma[~mask_te_drop] = False
        
        # --- Combine ---
        mask_tr = mask_tr_drop & mask_tr_ma
        mask_te = mask_te_drop & mask_te_ma
        
        feats_tr = feat_ma_tr
        feats_te = feat_ma_te
        
        # --- Logging ---
        logger.info(
            "  TripleDrop -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_tr.mean(), 100 * mask_te.mean() 
        )
        
        return feats_tr, feats_te, mask_tr, mask_te
