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

from scipy.optimize import differential_evolution

import logging
logger = logging.getLogger(__name__)

class StratMACDExtravaganza(BaseStrategy):
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
                
        opt_params["rf_active"]  = True #trial.suggest_categorical("rf_active", [True, False])
        
        opt_params["q_len_macd"]              = trial.suggest_float("q_len_macd", 0.15, 0.25)
        opt_params["q_len_macd"]              = np.clip(opt_params["q_len_macd"], 0.0, 1.0)
        
        opt_params["macd_n_fast"]             = trial.suggest_int("macd_n_fast", 1, 9)
        opt_params["macd_n_slow"]             = trial.suggest_int("macd_n_slow", 1, 2)
        opt_params["macd_n_sgnl"]             = trial.suggest_int("macd_n_sgnl", 1, 3)

        opt_params["catsample_max_lag"]          = trial.suggest_int("catsample_max_lag", 0, 55, step=5) # Note: try 100 at some point
        opt_params["max_features_colinsampling"] = trial.suggest_int("max_features_colinsampling", 10, 350, step=5)
        opt_params["threshold_colin_sampling"]   = trial.suggest_float("threshold_colin_sampling", 0.06, 0.3, log=True)
        
        opt_params["keep_perday_rf"]               = 3
        
        opt_params["LGB_num_boost_round"]           = 300 #trial.suggest_int("LGB_num_boost_round", 5, 455, step=5)
        opt_params["LGB_lambda_l1"]                 = 0.0005 #trial.suggest_float("LGB_lambda_l1", 0.00001, 0.1, log=True)
        opt_params["LGB_lambda_l2"]                 = 0.0 #trial.suggest_float("LGB_lambda_l2", 0.00001, 0.1, log=True)
        opt_params["LGB_feature_fraction"]          = 1.0 #trial.suggest_float("LGB_feature_fraction", 0.8, 1.0)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 75, 800, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 10, 50, step=5)
        opt_params["LGB_learning_rate"]             = trial.suggest_float("LGB_learning_rate", 0.007, 0.5, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 500, 2700, step=50)
        opt_params["LGB_min_gain_to_split"]         = 0.0 #trial.suggest_float("LGB_min_gain_to_split", 0.000001, 0.0005, log=True)
        opt_params["LGB_path_smooth"]               = 0.2 #trial.suggest_float("LGB_path_smooth", 0.01, 0.9, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = 0.2 #trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        opt_params["LGB_max_bin"]                   = 250 #trial.suggest_int("LGB_max_bin", 100, 800, step=25)
        opt_params["LGB_early_stopping_rounds"]     = opt_params["LGB_num_boost_round"]//10
        
        opt_params["inc_weight_features"]   = False
        if opt_params["inc_weight_features"]:
            opt_params["weight_wndw_ratio"]     = 0.015 #trial.suggest_float("weight_wndw_ratio", 0.01, 0.5, log=True)
            opt_params["min_weight"]            = 1.654769 #trial.suggest_float("min_weight", 0.1, 2.5, step=0.1)
            opt_params["trunc_weight_features"] = 3 #trial.suggest_int("trunc_weight_features", 1, 4)
            
        # --- SLTP gate ---
        opt_params["sltp_feats_imask"] = 0b01 #trial.suggest_categorical("sltp_feats_imask", [0b10, 0b11, 0b01])
        opt_params["sltp_n_clusters"] = 1 #trial.suggest_int("sltp_n_clusters", 1, 3)
        
        
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
        
        # --- MACD gate ---
        _, _, mask_tr, mask_te = self._compute_macd_feat(
            Xtr_time,
            Xte_time,
            ytr_opt,
            opt_params,
        )
        
        # -- log info ---
        logger.info(
            "  MACDEx -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_tr.mean(), 100 * mask_te.mean() 
        )
        n_dates_tr = meta_train.get_column('date').n_unique()
        n_dates_te = meta_test.get_column('date').n_unique()
        
        n_dates_masked_tr = meta_train.filter(pl.Series(mask_tr)).get_column('date').n_unique()
        n_dates_masked_te = meta_test.filter(pl.Series(mask_te)).get_column('date').n_unique()
        
        logger.info(f"  Dates kept -> train: {n_dates_masked_tr / n_dates_tr} | test: {n_dates_masked_te / n_dates_te}")
        
        # -- RF gate ---
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
        
        # weight features
        if opt_params["inc_weight_features"]:
            trunc_weights = opt_params.get("trunc_weight_features", 4)
            weightfeat_tr, weightfeat_te, wfeat_names = self._weight_features(Xtr_tree, Xte_tree, treenames, trunc=trunc_weights)

            ws: WeightSamples = WeightSamples(
                feat_train = weightfeat_tr[mask_tr], 
                y_train = ytr_opt, 
                treenames = wfeat_names,
                feat_test = weightfeat_te[mask_te], 
            )
            weights = ws.establish_weights(n_bin=15, min_bd=opt_params["min_weight"], wndw_ratio=opt_params["weight_wndw_ratio"])
        else:
            weights = None
        
        #LGBModel
        feat_tr_res = Xtr_tree[mask_tr][:, mask_tn]
        feat_te_res = Xte_tree[mask_te][:, mask_tn]
        y_lgb = ytr_opt[mask_tr]
        mm: MachineModels = MachineModels(opt_params)
        model, _ = mm.run_LGB(feat_tr_res, y_lgb, weights=weights)
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
        score_tr = scores_tr_rf
        score_te = scores_te_rf
        
        mask_tr_prelgb = mask_tr.copy()
        mask_te_prelgb = mask_te.copy()
        mask_tr[mask_tr] = mask_tr_rf
        mask_te[mask_te] = mask_te_rf
                
        # --- SL/TP matrices ---
        imask = opt_params["sltp_feats_imask"]
        n_clusters = opt_params["sltp_n_clusters"]
        Xtr_sltpfeats, Xte_sltpfeats = self._sltp_clust_features(
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
            Xtr_feats = Xtr_sltpfeats[mask_tr_prelgb],
            Xte_feats = Xte_sltpfeats[mask_te_prelgb],
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

        logger.info(f"  Precompute -> sl_tr {np.median(clust_sl_tr_mat, axis=0)} | tp_tr: {np.median(clust_tp_tr_mat, axis=0)}")

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
        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

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