import numpy as np
import optuna
import polars as pl

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

import logging
logger = logging.getLogger(__name__)

class StratSelectedMasks(BaseStrategy):
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
        opt_params["min_n_tar_daily"] = 30
        
        opt_params["macd_diff_qup"]                    = trial.suggest_float("macd_diff_qup", 0.86, 0.93)
        opt_params["macd_diff_qdown"]                  = trial.suggest_float("macd_diff_qdown", 0.985, 0.999)
        opt_params["macd_lookback_days"]               = 2000 # trial.suggest_int("macd_lookback_days", 1500, 2500)
        opt_params["macd_fast_period"]                 = trial.suggest_int("macd_fast_period", 8, 19)
        opt_params["macd_slow_period"]                 = trial.suggest_int("macd_slow_period", 25, 67)
        opt_params["macd_signal_period"]               = trial.suggest_int("macd_signal_period", 4, 12)
        opt_params["macd_fast_alpha"]                  = trial.suggest_float("macd_fast_alpha", 0.05, 0.25)
        opt_params["macd_slow_alpha"]                  = trial.suggest_float("macd_slow_alpha", 0.05, 0.25)
        opt_params["macd_signal_alpha"]                = trial.suggest_float("macd_signal_alpha", 0.05, 0.25)
        
        opt_params["atr_period"]                       = trial.suggest_int("atr_period", 5, 20)
        opt_params["atr_alpha"]                        = trial.suggest_float("atr_alpha", 0.05, 0.25)
        opt_params["atr_lookback_days"]                = 2000 # trial.suggest_int("atr_lookback_days", 1900, 2500)
        opt_params["atr_qup"]                          = trial.suggest_float("atr_qup", 0.85, 0.94)
        opt_params["atr_qdown"]                        = trial.suggest_float("atr_qdown", 0.980, 0.999)

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
        last_day_mask = meta_test["date"] == meta_test["date"].max()

        # --- MACD masks ---
        mask_train_macd, mask_test_macd = self._compute_macd_masks(
            Xtr_tree,
            Xte_tree,
            Xtr_time,
            Xte_time,
            meta_train,
            opt_params,
        )

        # --- ATR masks ---
        mask_train_atr, mask_test_atr = self._compute_atr_masks(
            Xtr_time,
            Xte_time,
            meta_train,
            opt_params,
        )

        # --- Extend MACD masks with ATR-only samples (union) ---
        mask_train = mask_train_macd | mask_train_atr
        mask_test = mask_test_macd | mask_test_atr
        
        mask_test = mask_test & last_day_mask.to_numpy()

        # If you later re-enable SL/TP optimization, use mask_train here:
        # sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
        #     ytr_tree[mask_train],
        #     ytr_tree_low[mask_train],
        #     ytr_tree_high[mask_train],
        #     ytr_tree_open[mask_train],
        #     n_grid=20,
        #     spread_cost=0.0,
        #     commission=0.0,
        # )

        sl_val, tp_val = 0.88, 2.0
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)
        
        score_tr = np.random.rand(Xtr_tree.shape[0])
        score_te = np.random.rand(Xte_tree.shape[0])

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