import numpy as np
import optuna
import polars as pl
import datetime
import lightgbm as lgb
import re

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import r_regression

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.hyperparameterTuning.HelperFunctions import HelperFunctions
from src.hyperparameterTuning.HelperMetrics import HelperMetrics
from src.hyperparameterTuning.HelperSLTP import HelperSLTP

from src.hyperparameterTuning.StratSelectedMasks import StratSelectedMasks

import logging
logger = logging.getLogger(__name__)

class StratLGBMOnFiltered(BaseStrategy):
    expected_load_params = {
        "idxAfterPrediction": 5,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }
    precompute_params = {
        "sl0" : 0.823295, 
        "sl1" : 0.833562, 
        "sl2" : 0.783671, 
        "sl3" : 0.878303, 
        "sl4" : 0.982187, 
    
        "tp0" : 1.307256, 
        "tp1" : 1.442779, 
        "tp2" : 1.377516, 
        "tp3" : 1.211979, 
        "tp4" : 1.378628, 
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
        
        opt_params["catsample_max_lag"] = trial.suggest_int("catsample_max_lag", 0, 50, step=5) # Note: try 100 at some point
        opt_params["max_features_colinsampling"] = trial.suggest_int("max_features_colinsampling", 25, 55, step=5)
        opt_params["threshold_colin_sampling"] = trial.suggest_float("threshold_colin_sampling", 0.05, 0.15, log=True)
        opt_params["inc_seasonal_in_colin"] = False #trial.suggest_categorical("inc_seasonal_in_colin", [True, False])
        
        opt_params["atr_period"]                       = trial.suggest_int("atr_period", 4, 7)
        opt_params["atr_alpha"]                        = trial.suggest_float("atr_alpha", 0.15, 0.20)
        opt_params["atr_qup"]                          = trial.suggest_float("atr_qup", 0.78, 0.85)
        opt_params["atr_qdown"]                        = 0.985 #trial.suggest_float("atr_qdown", 0.980, 0.999)
        
        opt_params["slope_period"]                     = trial.suggest_int("slope_period", 6, 10)
        opt_params["slope_qup"]                        = trial.suggest_float("slope_qup", 0.55, 0.65)
        opt_params["slope_qdown"]                      = 0.999 #trial.suggest_float("slope_qdown", 0.9, 0.999)
        
        opt_params["rmse_period"]                      = trial.suggest_int("rmse_period", 28, 35)
        opt_params["rmse_delay"]                       = trial.suggest_int("rmse_delay", 3, 5)
        opt_params["rmse_ma_wndw"]                     = trial.suggest_int("rmse_ma_wndw", 15, 20)
        opt_params["rmse_alpha"]                       = trial.suggest_float("rmse_alpha", 0.12, 0.17)
        opt_params["rmse_qup"]                         = trial.suggest_float("rmse_qup", 0.66, 0.74)
        opt_params["rmse_qdown"]                       = 0.985 #trial.suggest_float("rmse_qdown", 0.97, 0.999)
        
        opt_params["LGB_num_boost_round"]           = trial.suggest_int("LGB_num_boost_round", 25, 100, step=25)
        opt_params["LGB_lambda_l1"]                 = 0.0001#trial.suggest_float("LGB_lambda_l1", 0.00001, 0.05, log=True)
        opt_params["LGB_lambda_l2"]                 = 0.0001#trial.suggest_float("LGB_lambda_l2", 0.0001, 0.02, log=True)
        opt_params["LGB_feature_fraction"]          = 0.96 #trial.suggest_float("LGB_feature_fraction", 0.94, 0.99, log=True)
        opt_params["LGB_num_leaves"]                = trial.suggest_int("LGB_num_leaves", 300, 1200, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 4, 30, step=1)
        opt_params["LGB_learning_rate"]             = trial.suggest_float("LGB_learning_rate", 0.00001, 0.01, log=True)
        opt_params["LGB_min_data_in_leaf"]          = trial.suggest_int("LGB_min_data_in_leaf", 10, 700, step=25)
        opt_params["LGB_min_gain_to_split"]         = trial.suggest_float("LGB_min_gain_to_split", 0.0001, 0.9, log=True)
        opt_params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 0.01, 0.9, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = 0.2 #trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        opt_params["LGB_max_bin"]                   = 230 # trial.suggest_int("LGB_max_bin", 100, 800, step=25)
        opt_params["LGB_early_stopping_rounds"]     = opt_params["LGB_num_boost_round"]//10
        
        opt_params["do_transform"] = True #trial.suggest_categorical("do_transform", [True, False])
        opt_params["val_split"] = 0.01 #trial.suggest_float("val_split", 0.001, 0.05, log=True)

        #opt_params["ytree_kind"] = trial.suggest_categorical("ytree_kind", ["last", "abslast"]) #mean and max not very good
        opt_params["ytr_opt_alpha"] = 0.013285866056770094 #trial.suggest_float("ytr_opt_alpha", 0.01, 0.2, log=True)
        
        opt_params["sl0"] = 0.823295
        opt_params["sl1"] = 0.833562
        opt_params["sl2"] = 0.783671
        opt_params["sl3"] = 0.878303
        opt_params["sl4"] = 0.982187
        
        opt_params["tp0"] = 1.307256
        opt_params["tp1"] = 1.442779
        opt_params["tp2"] = 1.377516
        opt_params["tp3"] = 1.211979
        opt_params["tp4"] = 1.378628
        
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
        if meta_test.is_empty() or meta_test['date'].is_empty():
            return (
                np.zeros(Xtr_tree.shape[0], dtype=bool), 
                np.zeros(Xte_tree.shape[0], dtype=bool), 
                0.88*np.ones(Xtr_tree.shape[0], dtype=float),
                0.88*np.ones(Xte_tree.shape[0], dtype=float),
                2.0*np.ones(Xtr_tree.shape[0], dtype=float), 
                2.0*np.ones(Xte_tree.shape[0], dtype=float),
                np.ones(Xtr_tree.shape[0], dtype=float), 
                np.ones(Xte_tree.shape[0], dtype=float),
            )
        
        logger.info(f"  All test dates {meta_test.get_column('date').unique().to_list()}")
        
        # --- SETUP ---
        mm: MachineModels = MachineModels(opt_params)
        do_transform =      opt_params["do_transform"]
        val_split =         opt_params["val_split"]
        catsample_max_lag = opt_params.get("catsample_max_lag", 1)
        m_features = opt_params.get("max_features_colinsampling", 20)
        thr_colin = opt_params.get("threshold_colin_sampling", 0.2)
        inc_seasonal_in_colin = opt_params.get("inc_seasonal_in_colin", False)
        
        def to_time(x, inc_factor):
            return np.clip(np.tanh(np.log(np.clip(x, 1e-6, None)) * inc_factor) / 2.0 + 0.5, 1e-6, 1 - 1e-6)
        def to_tree(y, inc_factor):
            return np.exp(np.arctanh((y - 0.5) * 2.0)/inc_factor)    
        ytr_opt_alpha = opt_params.get("ytr_opt_alpha", 0.1)
        ytr_time_close = to_time(ytr_tree, 1.0)
        Xtr_time_close = Xtr_time[:, :, 0]
        close_comb = np.hstack((Xtr_time_close, ytr_time_close))
        ytr_opt = self._ema_2d(close_comb, alpha=ytr_opt_alpha)
        ytr_tree_opt = to_tree(ytr_opt[:, -1], 1.0)

        # --- Selected Masks ---
        mask_train, mask_test, _, _, _, _, _, _ = StratSelectedMasks().run(
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
            opt_params=opt_params,
        )
        
        logger.info(f"  Before masking: tr {Xtr_tree.shape}, te {Xte_tree.shape}")
        logger.info(f"    ytr_tree: n={ytr_tree.size}, mean={ytr_tree.mean():.6f}, std={ytr_tree.std():.6f}")
        Xtr_masked = Xtr_tree[mask_train]
        Xte_masked = Xte_tree[mask_test]
        yopt_masked = ytr_tree_opt[mask_train]
        logger.info(f"  After masking: tr {Xtr_masked.shape}, te {Xte_masked.shape}")
        logger.info(f"    yopt_masked: n={yopt_masked.size}, mean={yopt_masked.mean():.6f}, std={yopt_masked.std():.6f}")
        
        # --- FEATURE FILTERING ---
        tn = np.asarray(treenames, str)
        mask_tn = np.ones(tn.shape[0], dtype=bool)
        # Categorical subsampling of treenames
        mask_tn_goodlag, mask_tn_noncolin = self.catsample_treenames(
            treenames,
            max_lag=catsample_max_lag,
        )
        mask_tn &= mask_tn_goodlag | mask_tn_noncolin
        
        # Colinear sampling based on target correlation
        if inc_seasonal_in_colin:
            mask_tn_colin_input = mask_tn_goodlag
        else:
            mask_tn_colin_input = mask_tn_goodlag & ~mask_tn_noncolin
        mask_treenames_colinsampling = self.colinearity_treenames_mask(
            Xtr_masked[:, mask_tn_colin_input],
            yopt_masked,
            treenames=tn[mask_tn_colin_input],
            m_features=m_features,
            thr_crosscorr=thr_colin,
            row_ratio=1.0,
        )
        mask_tn[mask_tn_colin_input] = mask_treenames_colinsampling

        Xd_tr = Xtr_masked[:, mask_tn]
        Xd_te = Xte_masked[:, mask_tn]
        yd_tr = yopt_masked
        
        logger.info(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.info(f"    ytr_tree: n={yd_tr.size}, mean={yd_tr.mean():.6f}, std={yd_tr.std():.6f}")

        # --- MODEL TRAINING ---
        if do_transform:
            scaler = StandardScaler().fit(Xd_tr)
            Xd_tr = scaler.transform(Xd_tr)
            Xd_te = scaler.transform(Xd_te)

        sam_split = int(Xd_tr.shape[0] * (1 - val_split))
        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xd_tr[:sam_split],
                y_train=yd_tr[:sam_split],
                X_test=Xd_tr[sam_split:],
                y_test=yd_tr[sam_split:],
            )
        except Exception as e:
            logger.disabled = False
            logger.warning(f"  LGB failed: {e}")
            return 1.0
        finally:
            logger.disabled = False

        # LGB Predictions
        best_iter = getattr(model_lgb, "best_iteration", None)
        y_train_score = model_lgb.predict(Xd_tr, num_iteration=best_iter)
        y_test_score  = model_lgb.predict(Xd_te, num_iteration=best_iter)
        logger.info(f"  Test RMSE LGBM: {info['best_score']:.4f}")
        
        m = 5
        yfull_tr_score = np.zeros(Xtr_tree.shape[0], dtype=float)
        yfull_te_score = np.zeros(Xte_tree.shape[0], dtype=float)
        yfull_tr_score[mask_train] = y_train_score
        yfull_te_score[mask_test] = y_test_score
        meta_tr_pl_filtered = (
            meta_train.with_columns(
                pl.Series("prediction_ratio", yfull_tr_score)
            )
            .with_columns(
                pl.col("prediction_ratio")
                .rank(method="random", descending=True)
                .over("date")
                .alias("prediction_rank")
            )
        )
        meta_te_pl_filtered = (
            meta_test.with_columns(
                pl.Series("prediction_ratio", yfull_te_score)
            )
            .with_columns(
                pl.col("prediction_ratio")
                .rank(method="random", descending=True)
                .over("date")
                .alias("prediction_rank")
            )
        )
        
        mask_tr = meta_tr_pl_filtered["prediction_rank"].to_numpy() <= m
        mask_te = meta_te_pl_filtered["prediction_rank"].to_numpy() <= m
        
        mask_tr = mask_tr & mask_train
        mask_te = mask_te & mask_test
                
        logger.info(f"  Run -> train kept: {mask_tr.mean()} | test kept: {mask_te.mean()}")
        
        # --- SL/TP precomputation ---
        sl_vec = [opt_params["sl0"], opt_params["sl1"], opt_params["sl2"], opt_params["sl3"], opt_params["sl4"]]
        tp_vec = [opt_params["tp0"], opt_params["tp1"], opt_params["tp2"], opt_params["tp3"], opt_params["tp4"]]
        sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat = HelperSLTP.replicate(
            sl_vec,
            tp_vec,
            Xtr_tree,
            Xte_tree,
        )

        logger.info(f"  Precompute -> sl {sl_tr_mat[0,:]} | tp: {tp_tr_mat[0,:]}")

        return mask_tr, mask_te, sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat, yfull_tr_score, yfull_te_score

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
    ) -> tuple[np.ndarray, ...]:
        if treenames is None or meta_train is None or meta_test is None:
            raise ValueError("treenames, meta_train and meta_test are required.")

        # --- Extend MACD masks with ATR-only samples (union) ---
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
        
        params = self.precompute_params
        
        # --- SL/TP precomputation ---
        sl_vec = [params["sl0"], params["sl1"], params["sl2"], params["sl3"], params["sl4"]]
        tp_vec = [params["tp0"], params["tp1"], params["tp2"], params["tp3"], params["tp4"]]
        sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat = HelperSLTP.replicate(
            sl_vec,
            tp_vec,
            Xtr_tree,
            Xte_tree,
        )

        logger.info(
            "  Precompute -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(f"  Precompute -> sl {sl_tr_mat[0,:]} | tp: {tp_tr_mat[0,:]}")

        return mask_train, mask_test, sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------    
    def _ema_2d(
        self, 
        values: np.ndarray, 
        alpha: float
    ) -> np.ndarray:
        """
        values: (n_samples, n_steps)
        alpha: scalar EMA decay (0 < alpha <= 1)
        """
        out = np.empty_like(values, dtype=np.float64)
        out[:, 0] = values[:, 0]
        for t in range(1, values.shape[1]):
            out[:, t] = alpha * values[:, t] + (1.0 - alpha) * out[:, t - 1]
        return out
    
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
    
    def _subsample_rows(
        self,
        X: np.ndarray,
        y: np.ndarray,
        row_ratio: float,
        random_state: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optionally subsample rows of (X, y)."""
        if not (0 < row_ratio <= 1.0):
            raise ValueError("row_ratio must be in (0, 1].")

        if row_ratio == 1.0:
            return X, y

        n = X.shape[0]
        k = max(2, int(n * row_ratio))
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, k, replace=False)
        return X[idx], y[idx]
    
    def colinearity_treenames_mask(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treenames: np.ndarray,
        m_features: int,
        thr_crosscorr: float = 0.2,
        row_ratio: float = 1.0,
        random_state: int | None = None,
    ) -> np.ndarray:
        """
        Select up to m_features that are:
        - highly correlated with y
        - not too correlated (>|thr_crosscorr|) with already selected features.
        """
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of rows")

        # Optional row subsampling
        Xc, yc = self._subsample_rows(X, y, row_ratio, random_state)

        # Feature–feature correlation
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(Xc, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        corr[np.abs(corr) > 0.5] = 0.0

        # Feature–target correlations via sklearn
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_y = r_regression(Xc, yc)        # shape (p,)
        corr_y = np.nan_to_num(corr_y, nan=0.0)
        corr_y[np.abs(corr_y) > 0.5] = 0.0

        # Order by descending |corr(feature, y)|
        order = np.argsort(-np.abs(corr_y))
        keep = np.zeros(p, dtype=bool)

        # Greedy selection
        for j in order:
            if keep.sum() >= m_features:
                break
            if not keep.any() or np.all(np.abs(corr[j, keep]) <= thr_crosscorr):
                keep[j] = True
                logger.debug(f"Selected feature {treenames[j]} with |corr|={np.abs(corr_y[j])}")

        return keep