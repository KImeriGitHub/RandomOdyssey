"""
Momentum-based Optuna strategy aiming to outperform a buy-and-hold benchmark.

The strategy fits a ridge regression model on tree features to predict
risk-adjusted forward returns. It selects momentum signals based on
risk-aware thresholds and optimises stop-loss/take-profit levels from the
most confident trades. Hyperparameters controlling feature selection,
regularisation, and risk controls are tuned via Optuna.
"""

import logging
from typing import ClassVar

import numpy as np
import optuna
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

logger = logging.getLogger(__name__)


class StratOptunaMomentum(BaseStrategy):
    """Cross-sectional momentum model tuned with Optuna."""

    expected_load_params: ClassVar[dict] = {
        "idxAfterPrediction": 5,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params: ClassVar[dict] = {
        "default_sl": 0.94,
        "default_tp": 1.08,
        "baseline_idx_tar": 4,
    }

    base_params: ClassVar[dict] = {}

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        """Sample hyperparameters controlling model complexity and risk."""

        opt_params: dict = {}
        opt_params["idx_tar"] = trial.suggest_int("idx_tar", 3, 6)
        opt_params["ridge_alpha"] = trial.suggest_float(
            "ridge_alpha", 1e-4, 5.0, log=True
        )
        opt_params["max_features"] = trial.suggest_int("max_features", 25, 250, step=5)
        opt_params["top_quantile"] = trial.suggest_float("top_quantile", 0.05, 0.25)
        opt_params["risk_aversion"] = trial.suggest_float("risk_aversion", 0.2, 1.5)
        opt_params["min_signal"] = trial.suggest_float("min_signal", 0.0005, 0.01, log=True)
        opt_params["vol_floor"] = trial.suggest_float("vol_floor", 1e-4, 0.02, log=True)
        opt_params["sl_buffer"] = trial.suggest_float("sl_buffer", 0.01, 0.08)
        opt_params["tp_buffer"] = trial.suggest_float("tp_buffer", 0.02, 0.15)

        params = dict(self.base_params)
        params.update(opt_params)
        return params

    def run(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        ytr_tree_low: np.ndarray,
        ytr_tree_high: np.ndarray,
        ytr_tree_open: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame,
        meta_test: pl.DataFrame,
        opt_params: dict,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Fit a ridge regression momentum model and return masks, SL/TP, and scores.

        The model predicts log forward returns and penalizes volatility to focus on
        stable, higher-quality signals. Top-quantile signals are selected for
        trading, with stop-loss and take-profit optimized from the selected
        training subset.
        """

        idx_tar = opt_params.get("idx_tar", 4) - 1
        top_q = opt_params.get("top_quantile", 0.1)
        risk_aversion = opt_params.get("risk_aversion", 0.5)
        min_signal = opt_params.get("min_signal", 0.001)
        max_features = opt_params.get("max_features", Xtr_tree.shape[1])
        vol_floor = opt_params.get("vol_floor", 1e-3)
        sl_buffer = opt_params.get("sl_buffer", 0.03)
        tp_buffer = opt_params.get("tp_buffer", 0.07)

        # Target: log return on selected horizon
        target = np.log(np.clip(ytr_tree[:, idx_tar], 1e-6, None))

        # Feature selection by variance ranking to avoid noisy dimensions
        feature_var = np.var(Xtr_tree, axis=0)
        top_idx = np.argsort(feature_var)[::-1][:max_features]
        Xtr_sel = Xtr_tree[:, top_idx]
        Xte_sel = Xte_tree[:, top_idx]

        scaler = StandardScaler().fit(Xtr_sel)
        Xtr_scaled = scaler.transform(Xtr_sel)
        Xte_scaled = scaler.transform(Xte_sel)

        model = Ridge(alpha=opt_params.get("ridge_alpha", 0.1))
        model.fit(Xtr_scaled, target)
        pred_tr = model.predict(Xtr_scaled)
        pred_te = model.predict(Xte_scaled)

        # Volatility estimation from intraday high/low spread on target horizon
        vol_tr = np.maximum(ytr_tree_high[:, idx_tar] - ytr_tree_low[:, idx_tar], vol_floor)
        vol_te = vol_tr.mean() * np.ones_like(pred_te)

        risk_adj_tr = pred_tr / (vol_tr ** risk_aversion)
        risk_adj_te = pred_te / (vol_te ** risk_aversion)

        threshold = np.quantile(risk_adj_tr, 1 - top_q)
        mask_train = (risk_adj_tr >= threshold) & (pred_tr >= min_signal)
        mask_test = (risk_adj_te >= threshold) & (pred_te >= min_signal)

        if not np.any(mask_train):
            logger.info("No training signals passed the threshold; falling back to all samples.")
            mask_train = np.ones_like(mask_train, dtype=bool)

        # Optimize SL/TP using only confident signals
        sl_tr, tp_tr = self._compute_sl_tp(
            ytr_tree,
            ytr_tree_low,
            ytr_tree_high,
            ytr_tree_open,
            mask_train,
            idx_tar,
            sl_buffer,
            tp_buffer,
        )
        sl_te = np.full_like(pred_te, sl_tr)
        tp_te = np.full_like(pred_te, tp_tr)

        score_tr = risk_adj_tr
        score_te = risk_adj_te

        logger.info(
            "Momentum strategy -> kept %.2f%% train | %.2f%% test; threshold %.5f",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
            threshold,
        )

        return (
            mask_train,
            mask_test,
            np.full_like(pred_tr, sl_tr),
            sl_te,
            np.full_like(pred_tr, tp_tr),
            tp_te,
            score_tr,
            score_te,
        )

    def precompute(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        ytr_tree_low: np.ndarray,
        ytr_tree_high: np.ndarray,
        ytr_tree_open: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame,
        meta_test: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Baseline masks and SL/TP used prior to Optuna sampling."""

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        idx_tar = self.precompute_params.get("baseline_idx_tar", 4) - 1
        sl_default = float(self.precompute_params.get("default_sl", 0.94))
        tp_default = float(self.precompute_params.get("default_tp", 1.08))

        # Use data-driven stops if enough samples exist; otherwise fall back to defaults
        if Xtr_tree.shape[0] > 5:
            sl_opt, tp_opt, _ = HelperFunctions.optimize_sl_tp(
                ytr_tree[:, : idx_tar + 1],
                ytr_tree_low[:, : idx_tar + 1],
                ytr_tree_high[:, : idx_tar + 1],
                ytr_tree_open[:, : idx_tar + 1],
                n_grid=10,
                spread_cost=0.0005,
                commission=0.0000,
            )
            sl_default = float(sl_opt)
            tp_default = float(tp_opt)

        sl_tr = np.full(Xtr_tree.shape[0], sl_default, dtype=float)
        sl_te = np.full(Xte_tree.shape[0], sl_default, dtype=float)
        tp_tr = np.full(Xtr_tree.shape[0], tp_default, dtype=float)
        tp_te = np.full(Xte_tree.shape[0], tp_default, dtype=float)

        logger.info(
            "Precompute momentum -> sl %.4f | tp %.4f | mask %.2f%%/%.2f%%",
            sl_default,
            tp_default,
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_sl_tp(
        self,
        ytr_tree: np.ndarray,
        ytr_tree_low: np.ndarray,
        ytr_tree_high: np.ndarray,
        ytr_tree_open: np.ndarray,
        mask_train: np.ndarray,
        idx_tar: int,
        sl_buffer: float,
        tp_buffer: float,
    ) -> tuple[float, float]:
        """Derive stop-loss and take-profit tuned to the selected signals."""

        if np.sum(mask_train) < 10:
            logger.info("Too few selected samples for SL/TP optimisation; using defaults.")
            return float(self.precompute_params["default_sl"]), float(
                self.precompute_params["default_tp"]
            )

        arr = ytr_tree[mask_train][:, : idx_tar + 1]
        arr_low = ytr_tree_low[mask_train][:, : idx_tar + 1]
        arr_high = ytr_tree_high[mask_train][:, : idx_tar + 1]
        arr_open = ytr_tree_open[mask_train][:, : idx_tar + 1]

        sl_val, tp_val, info = HelperFunctions.optimize_sl_tp(
            arr,
            arr_low,
            arr_high,
            arr_open,
            n_grid=12,
            sl_max=max(self.precompute_params.get("default_sl", 0.94), 0.9 - sl_buffer),
            tp_min=min(self.precompute_params.get("default_tp", 1.08), 1.0 + tp_buffer),
            spread_cost=0.0005,
            commission=0.0000,
        )
        logger.info(
            "Optimised SL/TP -> sl %.4f | tp %.4f | info %s",
            sl_val,
            tp_val,
            info,
        )
        return float(sl_val), float(tp_val)