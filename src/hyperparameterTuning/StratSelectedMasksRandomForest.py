import numpy as np
import optuna
import polars as pl

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.hyperparameterTuning.HelperFunctions import HelperFunctions
from src.hyperparameterTuning.HelperSLTP import HelperSLTP
from sklearn.ensemble import RandomForestRegressor

import logging
logger = logging.getLogger(__name__)

class StratSelectedMasksRandomForest(BaseStrategy):
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
        
        opt_params["rf_gate_n_estimators"]        = trial.suggest_int("rf_gate_n_estimators", 101, 1001, step=100)
        opt_params["rf_gate_max_depth"]           = trial.suggest_int("rf_gate_max_depth", 3, 7)       
        opt_params["rf_gate_min_samples_split"]   = trial.suggest_int("rf_gate_min_samples_split", 10, 80, step=10)
        opt_params["rf_gate_min_samples_leaf"]    = trial.suggest_int("rf_gate_min_samples_leaf", 150, 550, step=10)
        opt_params["rf_gate_max_features"]        = 1.0 #trial.suggest_categorical("rf_gate_max_features", ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0])
        opt_params["rf_gate_bootstrap"]           = True #trial.suggest_categorical("rf_gate_bootstrap", [True, False])
        opt_params["rf_gate_oob_score"]           = True #trial.suggest_categorical("rf_gate_oob_score", [True, False])
        opt_params["rf_gate_n_jobs"]              = -1

        # --- RF gate: selection knobs ---
        opt_params["rf_gate_pred_q_low"]          = trial.suggest_float("rf_gate_pred_q_low", 0.88, 0.97)
        opt_params["rf_gate_pred_q_high"]         = 0.998 #trial.suggest_float("rf_gate_pred_q_high", 0.98, 0.999)

        # keep quantiles ordered (Optuna doesn't enforce)
        if opt_params["rf_gate_pred_q_low"] >= opt_params["rf_gate_pred_q_high"]:
            opt_params["rf_gate_pred_q_low"] = min(opt_params["rf_gate_pred_q_low"], opt_params["rf_gate_pred_q_high"] - 0.01)
        
        opt_params["atr_period"]                       = trial.suggest_int("atr_period", 2, 6)
        opt_params["atr_alpha"]                        = trial.suggest_float("atr_alpha", 0.13, 0.25)
        
        opt_params["slope_period"]                     = trial.suggest_int("slope_period", 4, 7)
        
        opt_params["rmse_period"]                      = trial.suggest_int("rmse_period", 11, 21)
        opt_params["rmse_delay"]                       = trial.suggest_int("rmse_delay", 3, 7)
        opt_params["rmse_ma_wndw"]                     = trial.suggest_int("rmse_ma_wndw", 11, 19)
        opt_params["rmse_alpha"]                       = trial.suggest_float("rmse_alpha", 0.10, 0.22)
        
        opt_params["sl0"] = 0.80 #trial.suggest_float("sl0", 0.80, 0.94)
        opt_params["sl1"] = 0.80 #trial.suggest_float("sl1", 0.80, 0.99)
        opt_params["sl2"] = 0.75 #trial.suggest_float("sl2", 0.72, 1.05)
        opt_params["sl3"] = 0.80 #trial.suggest_float("sl3", 0.70, 1.1)
        opt_params["sl4"] = 0.902187 #trial.suggest_float("sl4", 0.65, 1.2)
        
        opt_params["tp0"] = 1.407256 #trial.suggest_float("tp0", 1.1, 1.6)
        opt_params["tp1"] = 1.442779 #trial.suggest_float("tp1", 1.05, 1.8)
        opt_params["tp2"] = 1.477516 #trial.suggest_float("tp2", 1.00, 1.8)
        opt_params["tp3"] = 1.411979 #trial.suggest_float("tp3", 0.95, 1.9)
        opt_params["tp4"] = 1.478628 #trial.suggest_float("tp4", 0.9, 1.95)

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

        # --- ATR masks ---
        feat_train_atr, feat_test_atr = self._compute_atr_feat(
            Xtr_time,
            Xte_time,
            meta_train,
            opt_params,
        )
        
        # --- Slope masks ---
        feat_train_slope, feat_test_slope = self._compute_slope_feat(
            Xtr_time,
            Xte_time,
            meta_train,
            opt_params,
        )
        
        # --- RSME masks ---
        feat_train_rsme, feat_test_rsme = self._compute_rmse_feat(
            Xtr_time,
            Xte_time,
            meta_train,
            opt_params,
        )

        # --- Combine ---
        y_gate_tr = ytr_tree[:, -1]
        mask_train, mask_test = self._rf_gate_sample_masks(
            feat_train_atr=feat_train_atr,
            feat_train_slope=feat_train_slope,
            feat_train_rmse=feat_train_rsme,
            feat_test_atr=feat_test_atr,
            feat_test_slope=feat_test_slope,
            feat_test_rmse=feat_test_rsme,
            y_gate_tr=y_gate_tr,
            opt_params=opt_params,
        )
                
        sl_vec = [opt_params["sl0"], opt_params["sl1"], opt_params["sl2"], opt_params["sl3"], opt_params["sl4"]]
        tp_vec = [opt_params["tp0"], opt_params["tp1"], opt_params["tp2"], opt_params["tp3"], opt_params["tp4"]]
        sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat = HelperSLTP.replicate(
            sl_vec,
            tp_vec,
            Xtr_tree,
            Xte_tree,
        )

        logger.info(f"  Precompute -> sl {sl_tr_mat[0,:]} | tp: {tp_tr_mat[0,:]}")

        score_tr = np.ones(Xtr_tree.shape[0], dtype=np.float32)
        score_te = np.ones(Xte_tree.shape[0], dtype=np.float32)
        return mask_train, mask_test, sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat, score_tr, score_te

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

        #sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
        #    ytr_tree, 
        #    ytr_tree_low, 
        #    ytr_tree_high, 
        #    ytr_tree_open,
        #    n_grid=7,
        #    spread_cost=0.0,
        #    commission=0.0,
        #)
        sl_val, tp_val = 0.88, 2.0
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)
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
    def _rf_gate_sample_masks(
        self,
        feat_train_atr: np.ndarray,
        feat_train_slope: np.ndarray,
        feat_train_rmse: np.ndarray,
        feat_test_atr: np.ndarray,
        feat_test_slope: np.ndarray,
        feat_test_rmse: np.ndarray,
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
        Ztr = np.column_stack([feat_train_atr, feat_train_slope, feat_train_rmse])
        Zte = np.column_stack([feat_test_atr,  feat_test_slope,  feat_test_rmse])

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
        rf_gate.fit(Ztr, y_gate_tr)

        pred_tr = rf_gate.predict(Ztr)
        pred_te = rf_gate.predict(Zte)

        # --- knobs: prediction window ---
        pred_q_low  = opt_params.get("rf_gate_pred_q_low", 0.30)
        pred_q_high = opt_params.get("rf_gate_pred_q_high", 0.90)
        pred_lo, pred_hi = np.quantile(pred_tr, [pred_q_low, pred_q_high])

        mask_train = (
            (pred_tr >= pred_lo) & (pred_tr <= pred_hi)
        )

        mask_test = (
            (pred_te >= pred_lo) & (pred_te <= pred_hi)
        )

        return mask_train, mask_test
    
    def _ema_2d(
        self,
        values: np.ndarray,
        window: int,
        alpha: float,
    ) -> np.ndarray:
        """
        values: (n_samples, n_timesteps)
        alpha: scalar EMA decay (0 < alpha <= 1)
        """
        values = np.asarray(values, dtype=np.float64)
        n, T = values.shape
        out = np.empty((n, T), dtype=np.float64)

        decay = 1.0 - alpha

        for t in range(T):
            start = max(0, t - window + 1)
            x = values[:, start:t + 1]  # (n, L)
            L = x.shape[1]
            w = decay ** np.arange(L - 1, -1, -1, dtype=np.float64)  # newest gets weight 1
            w /= w.sum()
            out[:, t] = x @ w        
        return out
    
    def _sma_2d(
        self,
        values: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        True finite-window rolling SMA (like the truncated EMA pattern).
        Uses only the last `window` points.
        Warmup uses shorter windows.

        values: (n_samples, n_timesteps)
        """
        values = np.asarray(values, dtype=np.float64)
        n, T = values.shape
        out = np.empty((n, T), dtype=np.float64)

        for t in range(T):
            start = max(0, t - window + 1)
            out[:, t] = values[:, start:t + 1].mean(axis=1)
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
    
    def _compute_atr_feat(
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

        # --- Extract close / high / low ---
        close_tr = Xtr_time[:, :, 0].astype(np.float64, copy=False)
        high_tr  = Xtr_time[:, :, 2].astype(np.float64, copy=False)
        low_tr   = Xtr_time[:, :, 3].astype(np.float64, copy=False)

        close_te = Xte_time[:, :, 0].astype(np.float64, copy=False)
        high_te  = Xte_time[:, :, 2].astype(np.float64, copy=False)
        low_te   = Xte_time[:, :, 3].astype(np.float64, copy=False)

        # --- True Range and ATR (EMA of TR) ---
        tr_tr = self._true_range_2d(close_tr, high_tr, low_tr)
        tr_te = self._true_range_2d(close_te, high_te, low_te)

        atr_tr_full = self._ema_2d(tr_tr, window=atr_period, alpha=atr_alpha)
        atr_te_full = self._ema_2d(tr_te, window=atr_period, alpha=atr_alpha)

        # Use last time step per sample
        atr_tr_last = atr_tr_full[:, -1]
        atr_te_last = atr_te_full[:, -1]

        return atr_tr_last, atr_te_last
    
    def _compute_slope_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        meta_tr: pl.DataFrame,
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute masks based on slope of of price series in a certain window.
        """
        
        # --- Slope parameters ---
        slope_period = opt_params["slope_period"]
        
        # --- Extract close prices ---
        close_tr = Xtr_time[:, :, 0].astype(np.float64, copy=False)
        close_te = Xte_time[:, :, 0].astype(np.float64, copy=False)
        
        # --- Compute slopes via linear regression ---
        slope_tr = np.mean(np.diff(close_tr[:, -(slope_period+2):], axis=1), axis=1)  # shape (nS,)
        slope_te = np.mean(np.diff(close_te[:, -(slope_period+2):], axis=1), axis=1)  # shape (nS,)
        
        return slope_tr, slope_te
        
    def _compute_rmse_feat(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        meta_tr: pl.DataFrame,
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute masks based on RSME of price series in a certain window.
        """
        
        # --- RSME parameters ---
        rmse_period = opt_params["rmse_period"]
        rmse_ma_wndw = opt_params["rmse_ma_wndw"]
        rmse_delay = opt_params["rmse_delay"]
        rmse_alpha = opt_params["rmse_alpha"]
        
        # --- Extract close prices ---
        close_tr = Xtr_time[:, :, 0].astype(np.float64, copy=False)
        close_te = Xte_time[:, :, 0].astype(np.float64, copy=False)
        
        # --- Get MA ---
        ma_close_tr = self._ema_2d(close_tr, window=rmse_ma_wndw, alpha=rmse_alpha)
        ma_close_te = self._ema_2d(close_te, window=rmse_ma_wndw, alpha=rmse_alpha)
        
        # --- Compute RSME ---
        slice_tr = slice(-(rmse_period+2),-(rmse_delay)) if rmse_delay > 0 else slice(-(rmse_period+2), None)
        slice_te = slice_tr
        rmse_tr = np.sqrt(np.mean((
            close_tr[:, slice_tr] - ma_close_tr[:, slice_tr])**2, 
            axis=1
        ))
        rmse_te = np.sqrt(np.mean(
            (close_te[:, slice_te] - ma_close_te[:, slice_te])**2, 
            axis=1
        ))
        
        return rmse_tr, rmse_te
        