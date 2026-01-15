import logging
from typing import Tuple

import numpy as np
import optuna

from src.hyperparameterTuning.BaseStrategy import BaseStrategy

logger = logging.getLogger(__name__)


class StratOptunaMomentumVolatility(BaseStrategy):
    """Momentum-plus-volatility filter with Optuna-tuned risk controls.

    The strategy ranks securities by recent normalized momentum divided by
    realized volatility, favouring stable upward trends. Stop-loss and
    take-profit levels are volatility-aware to improve risk-adjusted
    performance and limit tail-risk. Hyperparameters are sampled via Optuna
    to adapt selection strength and risk buffers to the data regime.
    """

    expected_load_params = {
        "idxAfterPrediction": 5,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {}
    base_params = {}

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        """Sample risk-aware momentum parameters."""

        opt_params = {
            "momentum_window": trial.suggest_int("momentum_window", 3, 12),
            "vol_window": trial.suggest_int("vol_window", 5, 20),
            "min_n_tar_daily": trial.suggest_int("min_n_tar_daily", 5, 30),
            "top_quantile": trial.suggest_float("top_quantile", 0.6, 0.95),
            "vol_floor": trial.suggest_float("vol_floor", 1e-4, 5e-3, log=True),
            "sl_mult": trial.suggest_float("sl_mult", 0.8, 2.0),
            "tp_mult": trial.suggest_float("tp_mult", 1.0, 3.0),
            "idx_tar": trial.suggest_int("idx_tar", 3, 5),
        }

        params = dict(self.base_params)
        params.update(opt_params)
        return params

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _compute_momentum_vol(
        self,
        X_time: np.ndarray,
        timenames: list[str],
        momentum_window: int,
        vol_window: int,
        vol_floor: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (momentum, volatility, score, last_close) arrays for ranking."""

        if "FeatureLSTM_AdjClose" not in timenames:
            raise ValueError("Required time feature 'FeatureLSTM_AdjClose' missing.")

        idx_close = timenames.index("FeatureLSTM_AdjClose")
        close_series = X_time[:, :, idx_close]

        log_returns = np.diff(np.log(np.clip(close_series, 1e-8, None)), axis=1)

        momentum_window = min(momentum_window, log_returns.shape[1])
        vol_window = min(vol_window, log_returns.shape[1])

        recent_slice = slice(-momentum_window, None)
        mom = np.mean(log_returns[:, recent_slice], axis=1)

        vol_slice = slice(-vol_window, None)
        vol = np.std(log_returns[:, vol_slice], axis=1)
        vol = np.maximum(vol, vol_floor)

        last_close = close_series[:, -1]
        score = mom / vol
        return mom, vol, score, last_close

    def _volatility_stops(
        self,
        base_price: np.ndarray,
        vol: np.ndarray,
        sl_mult: float,
        tp_mult: float,
        spread_cost: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Derive SL/TP from volatility with sensible clipping."""

        tp = base_price * (1.0 + tp_mult * vol)
        sl = base_price * (1.0 - sl_mult * vol)

        sl = np.clip(sl, 0.90, 0.999)
        tp = np.clip(tp, 1.005, 3.0)

        if np.any(sl >= tp):
            buffer = 0.002 + spread_cost
            tp = np.maximum(tp, sl + buffer)

        return sl, tp

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply the risk-aware momentum selection to the test fold."""

        momentum_window = opt_params["momentum_window"]
        vol_window = opt_params["vol_window"]
        min_n_tar_daily = opt_params["min_n_tar_daily"]
        top_quantile = opt_params["top_quantile"]
        vol_floor = opt_params["vol_floor"]
        sl_mult = opt_params["sl_mult"]
        tp_mult = opt_params["tp_mult"]
        idx_tar = min(opt_params.get("idx_tar", ytr_tree.shape[1]), ytr_tree.shape[1])

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        _, vol_tr, score_tr, last_close_tr = self._compute_momentum_vol(
            Xtr_time, timenames, momentum_window, vol_window, vol_floor
        )
        mom_te, vol_te, score_te, last_close_te = self._compute_momentum_vol(
            Xte_time, timenames, momentum_window, vol_window, vol_floor
        )

        quantile_cut = np.quantile(score_te, top_quantile)
        mask_test = score_te >= quantile_cut

        n_dates_test = meta_test.get_column("date").n_unique()
        min_n_tar = min_n_tar_daily * n_dates_test
        if mask_test.sum() < min_n_tar:
            # backfill strongest remaining candidates to maintain diversification
            order = np.argsort(score_te)[::-1]
            need = min_n_tar - mask_test.sum()
            mask_test[order[:need]] = True

        sl_tr, tp_tr = self._volatility_stops(last_close_tr, vol_tr, sl_mult, tp_mult)
        sl_te, tp_te = self._volatility_stops(last_close_te, vol_te, sl_mult, tp_mult)

        logger.info(
            "  Run -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(
            "  Run -> quantile cutoff %.6f | mean momentum %.6f | mean vol %.6f",
            float(quantile_cut),
            float(np.mean(mom_te)),
            float(np.mean(vol_te)),
        )

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Default masks and conservative stops prior to Optuna search."""

        if timenames is None or meta_train is None or meta_test is None:
            raise ValueError("timenames, meta_train and meta_test are required.")

        # Use moderate defaults to allow a warm start before tuning
        defaults = {
            "momentum_window": 6,
            "vol_window": 10,
            "min_n_tar_daily": 10,
            "top_quantile": 0.8,
            "vol_floor": 1e-3,
            "sl_mult": 1.0,
            "tp_mult": 1.5,
            "idx_tar": min(4, ytr_tree.shape[1]),
        }

        mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te, _, _ = self.run(
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
            opt_params=defaults,
        )

        logger.info(
            "  Precompute -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te