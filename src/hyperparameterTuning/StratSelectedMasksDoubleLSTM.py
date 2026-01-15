import numpy as np
import optuna
import polars as pl

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.hyperparameterTuning.HelperFunctions import HelperFunctions
from src.hyperparameterTuning.HelperSLTP import HelperSLTP

from src.mathTools.RegimeChange import RegimeChange
from src.predictionModule.MachineModels import MachineModels

import logging
logger = logging.getLogger(__name__)

class StratSelectedMasksDoubleLSTM(BaseStrategy):
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
        self._device = "cuda"

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {
            "val_split":                trial.suggest_float("val_split", 0.01, 0.1, log=True),
            "t_win":                    trial.suggest_int("t_win", 5, 70, step=5),
            "LSTM_units":               16,
            "LSTM_num_layers":          1,
            "LSTM_learning_rate":       trial.suggest_float("LSTM_learning_rate", 2e-4, 1e-3, log=True),
            "LSTM_dropout":             0.05,
            "LSTM_inter_dropout":       0.05,
            "LSTM_recurrent_dropout":   0.05,
            "LSTM_epochs":              4,
            "LSTM_l1":                  0.001,
            "LSTM_l2":                  0.001,
            "LSTM_conv1d_kernel_size":  5,
            "min_n_tar":                25,
        }
        opt_params["atr_period"]                       = trial.suggest_int("atr_period", 2, 22)
        opt_params["atr_alpha"]                        = trial.suggest_float("atr_alpha", 0.03, 0.30)
        opt_params["atr_qup"]                          = trial.suggest_float("atr_qup", 0.65, 0.9)
        opt_params["atr_qdown"]                        = 0.99 #trial.suggest_float("atr_qdown", 0.980, 0.999)
        
        opt_params["slope_period"]                     = trial.suggest_int("slope_period", 2, 25)
        opt_params["slope_qup"]                        = trial.suggest_float("slope_qup", 0.5, 0.8)
        opt_params["slope_qdown"]                      = 1.0 #trial.suggest_float("slope_qdown", 0.9, 0.999)

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
        t_win = int(opt_params.get("t_win", 5))
        time_factor = 1.0
        val_split = float(opt_params.get("val_split", 0.1))
        
        # --- ATR masks ---
        mask_train_atr, mask_test_atr = self._compute_atr_masks(
            Xtr_time,
            Xte_time,
            meta_train,
            opt_params,
        )
        
        # --- Slope masks ---
        mask_train_slope, mask_test_slope = self._compute_slope_mask(
            Xtr_time,
            Xte_time,
            meta_train,
            opt_params,
        )

        # --- Combine ---
        mask_train = mask_train_atr & mask_train_slope
        mask_test = mask_test_atr & mask_test_slope
        
        # --- SL/TP matrices ---
        sl_vec, tp_vec = HelperSLTP.perfect_sl_tp(
            ytr_tree,
            ytr_tree_low,
            ytr_tree_high,
            ytr_tree_open,
            tp_buffer_pct=0.1,
            sl_buffer_pct=0.1,
        )
        sl_tar_time = RegimeChange.to_time(sl_vec, time_factor)
        tp_tar_time = RegimeChange.to_time(tp_vec, time_factor)
        
        N, T, F = Xtr_time.shape
        Xdtr_time = Xtr_time[:, -t_win:, :]
        Xdte_time = Xte_time[:, -t_win:, :]
        
        def default_res():
            m_tr = np.zeros(Xtr_time.shape[0], dtype=bool)
            m_te = np.zeros(Xte_time.shape[0], dtype=bool)
            sl_mat_tr = 0.80 * np.ones((Xtr_time.shape[0], ytr_tree.shape[1]), dtype=float)
            tp_mat_tr = 2.00 * np.ones((Xtr_time.shape[0], ytr_tree.shape[1]), dtype=float)
            sl_mat_te = 0.80 * np.ones((Xte_time.shape[0], ytr_tree.shape[1]), dtype=float)
            tp_mat_te = 1.50 * np.ones((Xte_time.shape[0], ytr_tree.shape[1]), dtype=float)
            score_tr = np.ones(Xtr_tree.shape[0], dtype=np.float32)
            score_te = np.ones(Xte_tree.shape[0], dtype=np.float32)
            return m_tr, m_te, sl_mat_tr, sl_mat_te, tp_mat_tr, tp_mat_te, score_tr, score_te

        mm = MachineModels(params=opt_params)
        try:
            val_split_n = max(1, int(N * (1-val_split)))
            model_sl, info_sl = mm.run_LSTM_torch(
                X_train=Xdtr_time[:val_split_n],
                y_train=sl_tar_time[:val_split_n],
                X_test=Xdtr_time[val_split_n:],
                y_test=sl_tar_time[val_split_n:],
                device=self._device,
                logger_disabled=True,
            )
            model_tp, info_tp = mm.run_LSTM_torch(
                X_train=Xdtr_time[:val_split_n],
                y_train=tp_tar_time[:val_split_n],
                X_test=Xdtr_time[val_split_n:],
                y_test=tp_tar_time[val_split_n:],
                device=self._device,
                logger_disabled=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[LSTM] training failed: %s", exc)
            return default_res()

        val_rmse_sl = float(info_sl.get("val_rmse", float("inf")))
        val_rmse_tp = float(info_tp.get("val_rmse", float("inf")))
        
        logger.debug("  LSTM val RMSE -> sl: %.6f | tp: %.6f", val_rmse_sl, val_rmse_tp)

        # sl predictions
        pred_tr_sl = mm.predict_LSTM_torch(model_sl, Xdtr_time, device=self._device)
        pred_te_sl = mm.predict_LSTM_torch(model_sl, Xdte_time, device=self._device)
        if pred_tr_sl.size == 0 or pred_te_sl.size == 0 or not np.all(np.isfinite(pred_te_sl)):
            logger.warning("[LSTM] sl predictions are empty or non-finite.")
            return default_res()
        pred_tr_sl_tree = RegimeChange.to_tree(pred_tr_sl, time_factor)
        pred_te_sl_tree = RegimeChange.to_tree(pred_te_sl, time_factor)
        
        logger.info("  sl preds stats: min %.4f | max %.4f | mean %.4f | std %.4f", np.min(pred_te_sl_tree), np.max(pred_te_sl_tree), np.mean(pred_te_sl_tree), np.std(pred_te_sl_tree))

        # tp predictions
        pred_tr_tp = mm.predict_LSTM_torch(model_tp, Xdtr_time, device=self._device)
        pred_te_tp = mm.predict_LSTM_torch(model_tp, Xdte_time, device=self._device)
        if pred_tr_tp.size == 0 or pred_te_tp.size == 0 or not np.all(np.isfinite(pred_tr_tp)):
            logger.warning("[LSTM] tp predictions are empty or non-finite.")
            return default_res()
        pred_tr_tp_tree = RegimeChange.to_tree(pred_tr_tp, time_factor)
        pred_te_tp_tree = RegimeChange.to_tree(pred_te_tp, time_factor)
        
        logger.info("  sl preds stats: min %.4f | max %.4f | mean %.4f | std %.4f", np.min(pred_te_tp_tree), np.max(pred_te_tp_tree), np.mean(pred_te_tp_tree), np.std(pred_te_tp_tree))
        
        #replicate
        sl_tr_mat = np.repeat(pred_tr_sl_tree[:, np.newaxis], ytr_tree.shape[1], axis=1)
        sl_te_mat = np.repeat(pred_te_sl_tree[:, np.newaxis], ytr_tree.shape[1], axis=1)
        tp_tr_mat = np.repeat(pred_tr_tp_tree[:, np.newaxis], ytr_tree.shape[1], axis=1)
        tp_te_mat = np.repeat(pred_te_tp_tree[:, np.newaxis], ytr_tree.shape[1], axis=1)
        
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

        atr_thr_qup = opt_params["atr_qup"]  # e.g. 0.7 for top 30% ATR
        atr_thr_qdown = opt_params["atr_qdown"]  # e.g. 0.0 for no lower bound

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

        atr_thr_qup = np.quantile(atr_tr_last, atr_thr_qup)
        atr_thr_qdown = np.quantile(atr_tr_last, atr_thr_qdown)

        # High-ATR regime
        mask_tr = (atr_tr_last >= atr_thr_qup) & (atr_tr_last <= atr_thr_qdown)
        mask_te = (atr_te_last >= atr_thr_qup) & (atr_te_last <= atr_thr_qdown)

        return mask_tr, mask_te
    
    def _compute_slope_mask(
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
        slope_qup = opt_params["slope_qup"]
        slope_qdown = opt_params["slope_qdown"]
        
        # --- Extract close prices ---
        close_tr = Xtr_time[:, :, 0].astype(np.float64, copy=False)
        close_te = Xte_time[:, :, 0].astype(np.float64, copy=False)
        
        # --- Compute slopes via linear regression ---
        slope_tr = np.mean(np.diff(close_tr[:, -(slope_period+2):], axis=1), axis=1)  # shape (nS,)
        slope_te = np.mean(np.diff(close_te[:, -(slope_period+2):], axis=1), axis=1)  # shape (nS,)
        
        slope_tr_thr_up, slope_tr_thr_down = np.quantile(slope_tr, [slope_qup, slope_qdown])
        
        mask_tr_slope = (slope_tr >= slope_tr_thr_up) & (slope_tr <= slope_tr_thr_down)
        mask_te_slope = (slope_te >= slope_tr_thr_up) & (slope_te <= slope_tr_thr_down)
        
        return mask_tr_slope, mask_te_slope
        
    def _compute_rmse_mask(
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
        rmse_qup = opt_params["rmse_qup"]
        rmse_qdown = opt_params["rmse_qdown"]
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
        
        rmse_tr_thr_up, rmse_tr_thr_down = np.quantile(rmse_tr, [rmse_qup, rmse_qdown])
        
        mask_tr_rsme = (rmse_tr >= rmse_tr_thr_up) & (rmse_tr <= rmse_tr_thr_down)
        mask_te_rsme = (rmse_te >= rmse_tr_thr_up) & (rmse_te <= rmse_tr_thr_down)
        
        return mask_tr_rsme, mask_te_rsme
        