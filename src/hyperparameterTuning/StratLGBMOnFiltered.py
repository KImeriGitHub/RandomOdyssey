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
from src.hyperparameterTuning.HelperSLTP import HelperSLTP
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
        "FilterSamples_cat_over10": False,
        "FilterSamples_cat_under5000": False,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.2": False,
        "FilterSamples_cat_volatility_qdown0.02": False,
        "FilterSamples_cat_volatility_qup0.975": False,
        "FilterSamples_cat_predictability_qup0.9": False,
        "FilterSamples_cat_predictability_qdown0.1": False,
        
        "macd_diff_qup"                    : 0.929618,
        "macd_diff_qdown"                  : 0.998680,
        "macd_lookback_days"               : 2000, # trial.suggest_int("macd_lookback_days", 1500, 2500)
        "macd_fast_period"                 : 12,
        "macd_slow_period"                 : 56,
        "macd_signal_period"               : 8,
        "macd_fast_alpha"                  : 0.146235,
        "macd_slow_alpha"                  : 0.088318,
        "macd_signal_alpha"                : 0.121511,
    
        "atr_period"                       : 13,
        "atr_alpha"                        : 0.176763,
        "atr_lookback_days"                : 2000, # trial.suggest_int("atr_lookback_days", 1900, 2500)
        "atr_qup"                          : 0.922995,
        "atr_qdown"                        : 0.991166,
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
        opt_params["inc_FeatureTA"] = True #trial.suggest_categorical("inc_FeatureTA", [True, False])
        opt_params["inc_GroupDynamics"] = True #trial.suggest_categorical("inc_GroupDynamics", [True, False])
        opt_params["inc_Categorical"] = True #trial.suggest_categorical("inc_Categorical", [True, False])
        opt_params["inc_Financials"] = True #trial.suggest_categorical("inc_Financials", [True, False])
        opt_params["inc_Mathematical"] = False #trial.suggest_categorical("inc_Mathematical", [True, False])
        opt_params["inc_Seasonal"] = True #trial.suggest_categorical("inc_Seasonal", [True, False])
        opt_params["exc_lag"] = True #trial.suggest_categorical("exc_lag", [True, False])
        
        opt_params["LGB_num_boost_round"]           = 100 #trial.suggest_int("LGB_num_boost_round", 25, 300, step=25)
        opt_params["LGB_lambda_l1"]                 = 0.0001#trial.suggest_float("LGB_lambda_l1", 0.00001, 0.05, log=True)
        opt_params["LGB_lambda_l2"]                 = 0.0001#trial.suggest_float("LGB_lambda_l2", 0.0001, 0.02, log=True)
        if opt_params.get("exc_lag"):
            opt_params["LGB_feature_fraction"]      = 0.96 #trial.suggest_float("LGB_feature_fraction", 0.94, 0.99, log=True)
        else:
            opt_params["LGB_feature_fraction"]      = 0.01 #trial.suggest_float("LGB_feature_fraction", 0.01, 0.1, log=True)
        opt_params["LGB_num_leaves"]                = 1050 #trial.suggest_int("LGB_num_leaves", 300, 1200, step=25)
        opt_params["LGB_max_depth"]                 = trial.suggest_int("LGB_max_depth", 4, 30, step=1)
        opt_params["LGB_learning_rate"]             = 0.000557353855828407 #trial.suggest_float("LGB_learning_rate", 0.00001, 0.01, log=True)
        opt_params["LGB_min_data_in_leaf"]          = 250 #trial.suggest_int("LGB_min_data_in_leaf", 100, 700, step=25)
        opt_params["LGB_min_gain_to_split"]         = 0.0051907347619269345 #trial.suggest_float("LGB_min_gain_to_split", 0.0001, 0.9, log=True)
        opt_params["LGB_path_smooth"]               = 0.6 #trial.suggest_float("LGB_path_smooth", 0.01, 0.9, log=True)
        opt_params["LGB_min_sum_hessian_in_leaf"]   = 0.2 #trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        opt_params["LGB_max_bin"]                   = 250 # trial.suggest_int("LGB_max_bin", 100, 800, step=25)
        opt_params["LGB_early_stopping_rounds"]     = opt_params["LGB_num_boost_round"]//10
        
        opt_params["do_transform"] = True #trial.suggest_categorical("do_transform", [True, False])
        opt_params["val_split"] = 0.01 #trial.suggest_float("val_split", 0.001, 0.05, log=True)

        #opt_params["ytree_kind"] = trial.suggest_categorical("ytree_kind", ["last", "abslast"]) #mean and max not very good
        opt_params["ytr_opt_alpha"] = 0.013285866056770094 #trial.suggest_float("ytr_opt_alpha", 0.01, 0.2, log=True)
        
        #opt_params["sl0"] = trial.suggest_float("sl0", 0.85, 0.89)
        #opt_params["sl1"] = trial.suggest_float("sl1", 0.80, 0.92)
        #opt_params["sl2"] = trial.suggest_float("sl2", 0.72, 0.8)
        #opt_params["sl3"] = trial.suggest_float("sl3", 0.76, 0.89)
        #opt_params["sl4"] = trial.suggest_float("sl4", 0.88, 1.09)
        #
        #opt_params["tp0"] = trial.suggest_float("tp0", 1.25, 1.6)
        #opt_params["tp1"] = trial.suggest_float("tp1", 1.24, 1.7)
        #opt_params["tp2"] = trial.suggest_float("tp2", 1.35, 1.4)
        #opt_params["tp3"] = trial.suggest_float("tp3", 1.17, 1.7)
        #opt_params["tp4"] = trial.suggest_float("tp4", 1.5, 2.3)
        
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
        logger.info(f"  Selecting last day only: {meta_test["date"].max()}")
        
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

        #if opt_params["ytree_kind"] == "last":
        #    ytr_tree_opt = ytr_tree[:, -1]
        #elif opt_params["ytree_kind"] == "mean":
        #    ytr_tree_opt = np.mean(ytr_tree, axis=1)
        #elif opt_params["ytree_kind"] == "abslast":
        #    ytr_tree_opt = np.abs(ytr_tree[:, -1])
        #elif opt_params["ytree_kind"] == "max":
        #    ytr_tree_opt = np.max(ytr_tree, axis=1)
            
        logger.info(f"  Before filtering: tr {Xtr_tree.shape}, te {Xte_tree.shape}")

        Xd_tr = Xtr_tree[:, mask_treenames]
        Xd_te = Xte_tree[:, mask_treenames]
        
        logger.info(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.info(f"   ytr_tree: n={ytr_tree_opt.size}, mean={ytr_tree_opt.mean():.6f}, std={ytr_tree_opt.std():.6f}")

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
        y_train_score = model_lgb.predict(Xd_tr, num_iteration=best_iter)
        y_test_score  = model_lgb.predict(Xd_te, num_iteration=best_iter)
        logger.info(f"  Test RMSE LGBM: {info['best_score']:.4f}")
        
        m = 5
        meta_tr_pl_filtered = (
            meta_train.with_columns(
                pl.Series("prediction_ratio", y_train_score)
            )
            .sort(["date", "prediction_ratio"], descending=[False, True])
            .with_columns(
                pl.col("prediction_ratio")
                .rank(method="random", descending=True)
                .over("date")
                .alias("prediction_rank")
            )
        )
        meta_te_pl_filtered = (
            meta_test.with_columns(
                pl.Series("prediction_ratio", y_test_score)
            )
            .sort(["date", "prediction_ratio"], descending=[False, True])
            .with_columns(
                pl.col("prediction_ratio")
                .rank(method="random", descending=True)
                .over("date")
                .alias("prediction_rank")
            )
        )
        
        mask_tr = meta_tr_pl_filtered["prediction_rank"].to_numpy() <= m
        mask_te = meta_te_pl_filtered["prediction_rank"].to_numpy() <= m
                
        logger.info(f"  Run -> train kept: {mask_tr.mean()} | test kept: {mask_te.mean()}")
        logger.info(f"  Run -> top scores test: {y_test_score[mask_te]} ")
        
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

        return mask_tr, mask_te, sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat, y_train_score, y_test_score

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

        params = self.precompute_params

        # --- MACD masks ---
        mask_train_macd, mask_test_macd = self._compute_macd_masks(
            Xtr_tree,
            Xte_tree,
            Xtr_time,
            Xte_time,
            meta_train,
            params,
        )

        # --- ATR masks ---
        mask_train_atr, mask_test_atr = self._compute_atr_masks(
            Xtr_time,
            Xte_time,
            meta_train,
            params,
        )

        # --- Extend MACD masks with ATR-only samples (union) ---
        mask_train = mask_train_macd | mask_train_atr
        mask_test = mask_test_macd | mask_test_atr
        
        # --- SL/TP precomputation ---
        sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat = HelperSLTP.calc_conditional(
            ytr_tree,
            ytr_tree_low,
            ytr_tree_high,
            ytr_tree_open,
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

    def _true_range_2d(
        self,
        close: np.ndarray,  # (n_samples, n_steps)
        high: np.ndarray,   # (n_samples, n_steps)
        low: np.ndarray,    # (n_samples, n_steps)
    ) -> np.ndarray:
        """
        Vectorized True Range for each sample and time step.
        TR_t = max(
            high_t - low_t,
            |high_t - close_{t-1}|,
            |low_t  - close_{t-1}|
        ), with TR_0 = high_0 - low_0
        """
        close = close.astype(np.float64, copy=False)
        high = high.astype(np.float64, copy=False)
        low = low.astype(np.float64, copy=False)

        tr = np.empty_like(close, dtype=np.float64)
        # t = 0
        tr[:, 0] = high[:, 0] - low[:, 0]

        for t in range(1, close.shape[1]):
            prev_close = close[:, t - 1]
            tr1 = high[:, t] - low[:, t]
            tr2 = np.abs(high[:, t] - prev_close)
            tr3 = np.abs(low[:, t] - prev_close)
            tr[:, t] = np.maximum(tr1, np.maximum(tr2, tr3))

        return tr

    def _compute_macd_masks(
        self,
        Xtr_tree: np.ndarray,          # kept for interface compatibility, not used
        Xte_tree: np.ndarray,          # kept for interface compatibility, not used
        Xtr_time: np.ndarray,          # (n_samples, 90, 5)
        Xte_time: np.ndarray,          # (n_samples, 90, 5)
        meta_tr: pl.DataFrame,
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recomputes MACD and signal from time-series tensors and builds masks.
        Only X*_time[..., 0] (first feature) is used as the price series.
        """

        # --- MACD parameters from opt_params ---
        fast_period = opt_params["macd_fast_period"]
        slow_period = opt_params["macd_slow_period"]
        signal_period = opt_params["macd_signal_period"]
        max_period = max(fast_period, slow_period, signal_period)+1  # +1 for day-start adjustments in the computations

        # Per-line decay rates (if not provided, use standard EMA from periods)
        alpha_fast = opt_params.get("macd_fast_alpha", 2.0 / (fast_period + 1.0))
        alpha_slow = opt_params.get("macd_slow_alpha", 2.0 / (slow_period + 1.0))
        alpha_signal = opt_params.get("macd_signal_alpha", 2.0 / (signal_period + 1.0))

        # --- Extract price series (first feature) ---
        # shape: (n_samples, n_steps)
        close_tr = Xtr_time[:,-max_period:, 0].astype(np.float64, copy=False)
        close_te = Xte_time[:,-max_period:, 0].astype(np.float64, copy=False)

        # --- EMA fast/slow and MACD for train/test ---
        ema_fast_tr = self._ema_2d(close_tr, alpha_fast)
        ema_slow_tr = self._ema_2d(close_tr, alpha_slow)
        macd_tr_full = ema_fast_tr - ema_slow_tr

        ema_fast_te = self._ema_2d(close_te, alpha_fast)
        ema_slow_te = self._ema_2d(close_te, alpha_slow)
        macd_te_full = ema_fast_te - ema_slow_te

        # --- Signal line (EMA of MACD) ---
        sig_tr_full = self._ema_2d(macd_tr_full, alpha_signal)
        sig_te_full = self._ema_2d(macd_te_full, alpha_signal)

        # Use only the last time step as the feature value for each sample
        macd_tr_last = macd_tr_full[:, -1]
        sig_tr_last = sig_tr_full[:, -1]
        macd_te_last = macd_te_full[:, -1]
        sig_te_last = sig_te_full[:, -1]

        # --- Lookback in terms of samples (same logic as before) ---
        lookback_days = opt_params["macd_lookback_days"]
        n_days = meta_tr.get_column("date").n_unique()
        days_ratio = np.clip(lookback_days / n_days, 0.0, 1.0)

        n_samples_lookback = int(np.ceil(days_ratio * Xtr_time.shape[0]))
        n_samples_lookback = int(np.clip(n_samples_lookback, 1, Xtr_time.shape[0]))

        # Quantiles are computed from the last n_samples_lookback train samples
        diff_tr_lookback = macd_tr_last[-n_samples_lookback:] - sig_tr_last[-n_samples_lookback:]

        qdown = opt_params["macd_diff_qdown"]
        qup = opt_params["macd_diff_qup"]
        qup_val = np.quantile(diff_tr_lookback, qup)
        qdown_val = np.quantile(diff_tr_lookback, qdown)

        # --- Apply thresholds to all samples ---
        diff_tr_all = macd_tr_last - sig_tr_last
        diff_te_all = macd_te_last - sig_te_last

        # NOTE: It is that qdown > qup , hence [qup_val, qdown_val] interval
        mask_tr = (diff_tr_all <= qdown_val) & (diff_tr_all >= qup_val)
        mask_te = (diff_te_all <= qdown_val) & (diff_te_all >= qup_val)

        return mask_tr, mask_te
    
    def _compute_atr_masks(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        meta_tr: pl.DataFrame,
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds train/test masks based on Average True Range (ATR).
        Uses features:
            0: AdjClose
            2: AdjHigh
            3: AdjLow
        Threshold is a quantile of ATR over a lookback on train.
        """

        # --- ATR parameters ---
        atr_period = opt_params["atr_period"]
        atr_alpha = opt_params.get("atr_alpha", 2.0 / (atr_period + 1.0))

        # Re-use MACD lookback unless an ATR-specific one is provided
        lookback_days = opt_params.get("atr_lookback_days")
        atr_qup = opt_params["atr_qup"]  # e.g. 0.7 for top 30% ATR
        atr_qdown = opt_params["atr_qdown"]  # e.g. 0.0 for no lower bound

        # --- Extract close / high / low ---
        close_tr = Xtr_time[:, -(atr_period+1):, 0].astype(np.float64, copy=False)
        high_tr = Xtr_time[:, -(atr_period+1):, 2].astype(np.float64, copy=False)
        low_tr  = Xtr_time[:, -(atr_period+1):, 3].astype(np.float64, copy=False)

        close_te = Xte_time[:, -(atr_period+1):, 0].astype(np.float64, copy=False)
        high_te = Xte_time[:, -(atr_period+1):, 2].astype(np.float64, copy=False)
        low_te  = Xte_time[:, -(atr_period+1):, 3].astype(np.float64, copy=False)

        # --- True Range and ATR (EMA of TR) ---
        tr_tr = self._true_range_2d(close_tr, high_tr, low_tr)
        tr_te = self._true_range_2d(close_te, high_te, low_te)

        atr_tr_full = self._ema_2d(tr_tr, atr_alpha)
        atr_te_full = self._ema_2d(tr_te, atr_alpha)

        # Use last time step per sample
        atr_tr_last = atr_tr_full[:, -1]
        atr_te_last = atr_te_full[:, -1]

        # --- Lookback in terms of samples (same logic as MACD) ---
        n_days = meta_tr.get_column("date").n_unique()
        days_ratio = np.clip(lookback_days / n_days, 0.0, 1.0)

        n_samples_lookback = int(np.ceil(days_ratio * Xtr_time.shape[0]))
        n_samples_lookback = int(np.clip(n_samples_lookback, 1, Xtr_time.shape[0]))

        atr_tr_lookback = atr_tr_last[-n_samples_lookback:]
        atr_qup = np.quantile(atr_tr_lookback, atr_qup)
        atr_qdown = np.quantile(atr_tr_lookback, atr_qdown)

        # High-ATR regime
        mask_tr = (atr_tr_last >= atr_qup) & (atr_tr_last <= atr_qdown)
        mask_te = (atr_te_last >= atr_qup) & (atr_te_last <= atr_qdown)

        return mask_tr, mask_te