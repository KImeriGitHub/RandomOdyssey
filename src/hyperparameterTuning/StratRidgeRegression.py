import numpy as np
import optuna
import polars as pl
from typing import Any

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples

import logging
logger = logging.getLogger(__name__)


class StratRidgeRegression(BaseStrategy):
    """Simple ridge-regression strategy targeting market outperformance.

    The strategy fits a lightweight ridge model on the tree features in order
    to obtain a continuous ranking for each sample.  During evaluation we keep
    the most attractive candidates per-date and compare the geometric mean of
    their realised returns against the geometric mean of the whole universe
    (a simple market proxy).  Optuna is then used to tune the ridge
    regularisation strength, feature subset size and selection aggressiveness.
    """
    expected_load_params = {
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }
    precompute_params = {
        "min_close": 20.0,
    }

    base_params = {}

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        params = dict(self.base_params)
        params.update({
            "ridge_alpha":          trial.suggest_float("ridge_alpha", 1e-4, 10.0, log=True),
            "feature_fraction":     trial.suggest_float("feature_fraction", 0.2, 1.0),
            "scale_inputs":         trial.suggest_categorical("scale_inputs", [True, False]),
            "selection_quantile":   trial.suggest_float("selection_quantile", 0.6, 0.98),
            "min_selection":        trial.suggest_int("min_selection", 5, 50),
            "min_daily_pool":       trial.suggest_int("min_daily_pool", 1, 5),
            "price_floor":          trial.suggest_float("price_floor", 2.0, 40.0),
        })
        return params

    def score(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        yte_tree: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame | None,
        meta_test: pl.DataFrame | None,
        opt_params: dict[str, Any],
    ) -> float:
        del Xtr_time, Xte_time, treenames, timenames  # not used by this strategy

        if Xtr_tree.size == 0 or Xte_tree.size == 0:
            logger.warning("StratBeatMarket received empty feature arrays.")
            return 1.0

        price_floor = float(opt_params.get("price_floor", 0.0) or 0.0)

        (
            Xtr_tree,
            ytr_tree,
            meta_train,
        ) = self._apply_price_filter(Xtr_tree, ytr_tree, meta_train, price_floor)
        (
            Xte_tree,
            yte_tree,
            meta_test,
        ) = self._apply_price_filter(Xte_tree, yte_tree, meta_test, price_floor)

        if Xtr_tree.size == 0 or Xte_tree.size == 0:
            logger.warning("StratBeatMarket filters removed all samples.")
            return 1.0

        X_train, prep = self._prepare_features(
            Xtr_tree,
            feature_fraction=float(opt_params["feature_fraction"]),
            scale_inputs=bool(opt_params["scale_inputs"]),
        )
        X_test = self._transform_features(Xte_tree, prep)

        try:
            ridge_weights, ridge_bias = self._fit_ridge(
                X_train, np.asarray(ytr_tree, dtype=float), float(opt_params["ridge_alpha"])
            )
        except np.linalg.LinAlgError as exc:
            logger.warning("StratBeatMarket ridge solve failed: %s", exc)
            return 1.0

        preds = X_test @ ridge_weights + ridge_bias

        date_series = self._extract_dates(meta_test)
        selection_mask = self._select_candidates(
            preds,
            date_series,
            selection_quantile=float(opt_params["selection_quantile"]),
            min_daily_pool=int(opt_params["min_daily_pool"]),
            min_selection=int(opt_params["min_selection"]),
        )

        if not selection_mask.any():
            logger.warning("StratBeatMarket failed to select any candidates.")
            return 1.0

        selected_returns = np.asarray(yte_tree, dtype=float)[selection_mask]
        selected_dates = date_series[selection_mask] if date_series is not None else None

        portfolio_return = self._geometric_mean_by_date(selected_returns, selected_dates)
        if not np.isfinite(portfolio_return) or portfolio_return <= 0:
            logger.warning("StratBeatMarket produced invalid portfolio return: %s", portfolio_return)
            return 1.0

        all_dates = date_series if date_series is not None else None
        baseline_return = self._geometric_mean_by_date(np.asarray(yte_tree, dtype=float), all_dates)
        if not np.isfinite(baseline_return) or baseline_return <= 0:
            logger.info("Baseline return invalid; using raw portfolio return.")
            return float(portfolio_return)

        score = portfolio_return
        logger.info(
            "BeatMarket score: %.5f | portfolio_return=%.5f | baseline_return=%.5f | selected=%d/%d",
            score,
            portfolio_return,
            baseline_return,
            int(selection_mask.sum()),
            selection_mask.size,
        )

        if not np.isfinite(score) or score <= 0:
            return 1.0
        return float(score)

    def mask_precompute(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        yte_tree: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame | None,
        meta_test: pl.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del Xtr_time, ytr_tree, Xte_time, yte_tree, treenames, timenames

        params = self.precompute_params.copy()

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        min_close = params.get("min_close")
        threshold = float(min_close)
        fs_params = params
        fs_params[f"FilterSamples_cat_over{threshold: .2f}"] = True
        fs_params[f"FilterSamples_cat_under2000.0"] = True

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree,
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=yte_tree,
            meta_train=meta_train,
            meta_test=meta_test,
            params=params,
        )
        cat_train, cat_test = fs.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test

        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )

        return mask_train, mask_test

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_price_filter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pl.DataFrame | None,
        floor: float,
    ) -> tuple[np.ndarray, np.ndarray, pl.DataFrame | None]:
        if meta is None or "Close" not in meta.columns or floor <= 0:
            return X, y, meta

        mask = meta["Close"].fill_null(0.0).to_numpy() >= floor
        if mask.shape[0] != X.shape[0]:
            logger.warning("Mismatch between metadata and feature rows; skipping price filter.")
            return X, y, meta

        if mask.all():
            return X, y, meta

        X_filtered = X[mask]
        y_filtered = y[mask]
        meta_filtered = meta.filter(pl.Series(mask))
        return X_filtered, y_filtered, meta_filtered

    def _prepare_features(
        self,
        X: np.ndarray,
        feature_fraction: float,
        scale_inputs: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n_samples, n_features = X.shape
        n_keep = max(1, int(np.ceil(n_features * float(feature_fraction))))
        variances = np.var(X, axis=0)
        keep_idx = np.argsort(variances)[-n_keep:]
        keep_idx.sort()
        X_sel = X[:, keep_idx]

        prep: dict[str, Any] = {"indices": keep_idx}
        if scale_inputs:
            mean = X_sel.mean(axis=0)
            std = X_sel.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            X_sel = (X_sel - mean) / std
            prep["mean"] = mean
            prep["std"] = std
        else:
            prep["mean"] = None
            prep["std"] = None

        return X_sel, prep

    @staticmethod
    def _transform_features(X: np.ndarray, prep: dict[str, Any]) -> np.ndarray:
        keep_idx = prep["indices"]
        X_sel = X[:, keep_idx]
        mean = prep.get("mean")
        std = prep.get("std")
        if mean is not None and std is not None:
            X_sel = (X_sel - mean) / std
        return X_sel

    @staticmethod
    def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
        if X.ndim != 2:
            raise ValueError("Feature matrix must be 2-dimensional.")
        if y.ndim != 1:
            y = y.ravel()

        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X_aug = np.hstack([X, ones])
        gram = X_aug.T @ X_aug
        reg = np.eye(gram.shape[0])
        reg[-1, -1] = 0.0  # do not regularise the bias term
        coeffs = np.linalg.solve(gram + alpha * reg, X_aug.T @ y)
        weights = coeffs[:-1]
        bias = float(coeffs[-1])
        return weights, bias

    @staticmethod
    def _extract_dates(meta: pl.DataFrame | None) -> np.ndarray | None:
        if meta is None or "date" not in meta.columns:
            return None
        dates = meta["date"].to_numpy()
        return dates

    @staticmethod
    def _select_candidates(
        preds: np.ndarray,
        dates: np.ndarray | None,
        *,
        selection_quantile: float,
        min_daily_pool: int,
        min_selection: int,
    ) -> np.ndarray:
        n = preds.shape[0]
        selection_mask = np.zeros(n, dtype=bool)

        if dates is None:
            threshold = np.quantile(preds, selection_quantile)
            local_mask = preds >= threshold
            if not local_mask.any():
                top_idx = np.argsort(preds)[-max(1, min_selection):]
                selection_mask[top_idx] = True
            else:
                selection_mask = local_mask
        else:
            unique_dates = np.unique(dates)
            for date in unique_dates:
                idx = np.where(dates == date)[0]
                if idx.size == 0:
                    continue
                threshold = np.quantile(preds[idx], selection_quantile)
                local_mask = preds[idx] >= threshold
                if local_mask.sum() < min_daily_pool:
                    k = min(idx.size, max(min_daily_pool, 1))
                    top_idx = idx[np.argsort(preds[idx])[-k:]]
                    selection_mask[top_idx] = True
                else:
                    selection_mask[idx[local_mask]] = True

        total_selected = int(selection_mask.sum())
        if total_selected == 0:
            k = min(n, max(1, min_selection))
            top_idx = np.argsort(preds)[-k:]
            selection_mask[top_idx] = True
        elif total_selected < min_selection:
            mask = np.zeros(n, dtype=bool)
            k = min(n, max(1, min_selection))
            top_idx = np.argsort(preds)[-k:]
            mask[top_idx] = True
            selection_mask = mask

        return selection_mask

    @staticmethod
    def _geometric_mean_by_date(values: np.ndarray, dates: np.ndarray | None) -> float:
        vals = np.asarray(values, dtype=float)
        vals = np.clip(vals, 1e-9, None)
        log_vals = np.log(vals)
        if vals.size == 0:
            return float("nan")

        if dates is None:
            return float(np.exp(log_vals.mean()))

        daily_means: list[float] = []
        for date in np.unique(dates):
            mask = dates == date
            if not np.any(mask):
                continue
            daily_means.append(log_vals[mask].mean())

        if not daily_means:
            return float("nan")
        return float(np.exp(np.mean(daily_means)))