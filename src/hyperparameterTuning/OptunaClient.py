import datetime
import logging
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import optuna

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta
from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.LoadupSamples import LoadupSamples
from src.predictionModule.MachineModels import MachineModels as _MachineModels

logger = logging.getLogger(__name__)

SlicePair = Tuple[slice, slice]

class OptunaClient:
    """Utility orchestrating Optuna optimisation on top of ``LoadupSamples``."""

    def __init__(
        self,
        ls: LoadupSamples,
        n_splits: int,
        n_test_days: int,
        n_training_days: int,
        *,
        rng: Optional[random.Random] = None,
        model_cls: Optional[type] = None,
    ) -> None:
        self._extract_training_arrays(ls)
        self.n_splits = n_splits
        self.n_test_days = n_test_days
        self.n_training_days = n_training_days
        self._rng = rng if rng is not None else random.Random()
        if model_cls is None:
            model_cls = _MachineModels
        self._model_cls = model_cls

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------
    def get_split_dates(
        self,
        final_split_date: datetime.date,
        start_train_date: datetime.date,
    ) -> List[datetime.date]:
        """Return random split dates respecting the provided constraints."""

        start = start_train_date + datetime.timedelta(days=self.n_training_days - 1)
        end = final_split_date - datetime.timedelta(days=self.n_test_days)
        if start > end:
            raise ValueError("Training window does not fit between start and end dates.")

        eligible = [
            start + datetime.timedelta(days=i)
            for i in range((end - start).days + 1)
            if (start + datetime.timedelta(days=i)).weekday() < 5
        ]
        if len(eligible) < self.n_splits:
            raise ValueError(
                f"Too few eligible split dates ({len(eligible)}) for n_splits={self.n_splits}."
            )

        split_dates = sorted(self._rng.sample(eligible, self.n_splits))

        for date in split_dates:
            logger.info("  Split date: %s", date)
        logger.info("  n_reruns: %s", self.n_splits)
        logger.info("  max_training_days: %s", self.n_training_days)
        logger.info("  n_testdays: %s", self.n_test_days)
        logger.info("  start_train_date: %s", start_train_date)
        logger.info("  final_split_date: %s", final_split_date)

        return split_dates

    def get_pivots(self) -> List[int]:
        """Sample pivot indices (last training day) used for slicing."""

        dates_tr = self.meta_train["date"].unique().sort()
        dates_tr_idx = dfta(self.meta_train, "date").getNextLowerOrEqualIndices(dates_tr)

        N = len(dates_tr_idx)
        lo = self.n_training_days
        hi = (N - 1) - (self.n_test_days)
        eligible = list(range(lo, hi + 1))
        if len(eligible) < self.n_splits:
            raise ValueError(
                f"Too few eligible pivots ({len(eligible)}) for n_splits={self.n_splits}."
            )

        bases = np.linspace(lo, hi, self.n_splits).round().astype(int)
        gap = (hi - lo) / (self.n_splits - 1)

        # jitter by one fourth gap (integer), seeded from self._rng for reproducibility
        jmax = int(gap // 4)
        rng = np.random.default_rng(self._rng.getrandbits(64))
        jitter = rng.integers(-jmax, jmax + 1, size=self.n_splits) if jmax > 0 else 0

        pivots = np.clip(bases + jitter, lo, hi).astype(int)
        pivots = [int(i) for i in pivots]
        for p in pivots:
            logger.info("  Pivot %s: Date %s", p, dates_tr[p])
        return pivots

    def get_slices(self) -> List[SlicePair]:
        """Return list of (train_slice, test_slice) pairs for cross-validation."""
        
        pivots = self.get_pivots()
        dates_tr = self.meta_train["date"].unique().sort()
        dates_tr_idx = dfta(self.meta_train, "date").getNextLowerOrEqualIndices(dates_tr)
        N = len(dates_tr_idx)

        slices: List[SlicePair] = [None] * self.n_splits  # type: ignore[list-item]
        for i, p in enumerate(pivots):
            if p + self.n_test_days + 1 > N:
                raise ValueError("Pivot too far to the end of training dates.")

            tr_l_idx = dates_tr_idx[p - self.n_training_days + 1]
            tr_u_idx = dates_tr_idx[p + 1] - 1
            te_l_idx = dates_tr_idx[p + 1]
            if p + self.n_test_days + 1 == N:
                te_u_idx = len(self.meta_train["date"]) - 1
            else:
                te_u_idx = dates_tr_idx[p + self.n_test_days + 1] - 1

            s_tr = slice(tr_l_idx, tr_u_idx + 1)
            s_te = slice(te_l_idx, te_u_idx + 1)
            slices[i] = (s_tr, s_te)

        return slices

    # ------------------------------------------------------------------
    # Objective helper
    # ------------------------------------------------------------------
    def make_objective(self, strategy: BaseStrategy) -> Callable[[optuna.Trial], float]:
        """Create an Optuna objective callable using the provided strategy."""

        slices = self.get_slices()
        
        ####################
        ## PRE-PROCESSING ##
        ####################
        preprocess_masks = [None] * self.n_splits
        scores = [None] * self.n_splits
        for i, (s_tr, s_te) in enumerate(slices):
            Xtr_tree = self.X_tree[s_tr].copy()
            Xtr_time = self.X_time[s_tr].copy()
            ytr_tree = self.y_tree[s_tr].copy()
            Xte_tree = self.X_tree[s_te].copy()
            Xte_time = self.X_time[s_te].copy()
            yte_tree = self.y_tree[s_te].copy()

            meta_train_slice = self.meta_train[s_tr] if self.meta_train is not None else None
            meta_test_slice = self.meta_train[s_te] if self.meta_train is not None else None

            try:
                mask_train_pre, mask_test_pre = strategy.mask_precompute(
                    Xtr_tree,
                    Xtr_time,
                    ytr_tree,
                    Xte_tree,
                    Xte_time,
                    yte_tree,
                    treenames=self.treenames,
                    timenames=self.timenames,
                    meta_train=meta_train_slice,
                    meta_test=meta_test_slice,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("mask_precompute failed: %s", exc)
                full_mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
                full_mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
                mask_train_pre, mask_test_pre = full_mask_train, full_mask_test

            preprocess_masks[i] = (mask_train_pre, mask_test_pre)
            scores[i] = np.exp(np.mean(np.log(yte_tree[mask_test_pre])))
            
        scores = [float(s) if s is not None and np.isfinite(s) else 1.0 for s in scores]
        logger.info("Preprocessing complete.")
        logger.info("Precomputed scores per split (geometric mean of y_test): %s", scores)
        logger.info("Precomputed geometric mean of scores: %s", float(np.exp(np.mean(np.log(np.array(scores))))))
        logger.info("Precomputed arithmetic mean of scores: %s", float(np.mean(np.array(scores))))
        logger.info("Precomputed variance of scores: %s", float(np.var(np.array(scores))))
        logger.info("Precomputed standard deviation of scores: %s", float(np.std(np.array(scores))))
        logger.info("Precomputed min score: %s", float(np.min(np.array(scores))))
        logger.info("Precomputed max score: %s", float(np.max(np.array(scores))))
        logger.info("Precomputed Sharpe ratio (mean/std): %s", (float(np.mean(np.array(scores))) - 1.0) / np.std(np.array(scores)))

        ######################
        ## Objective function
        ######################
        def objective(trial: optuna.Trial) -> float:
            opt_params = strategy.sample_params(trial)
            logger.info("Trial %s with params: %s", trial.number, opt_params)

            scores = []
            n_valid = 0
            for i, (s_tr, s_te) in enumerate(slices):
                mask_train_pre, mask_test_pre = preprocess_masks[i]

                Xtr_tree = self.X_tree[s_tr][mask_train_pre]
                ytr_tree = self.y_tree[s_tr][mask_train_pre]
                Xte_tree = self.X_tree[s_te][mask_test_pre]
                yte_tree = self.y_tree[s_te][mask_test_pre]
                Xtr_time = self.X_time[s_tr][mask_train_pre]
                Xte_time = self.X_time[s_te][mask_test_pre]

                meta_tr = self.meta_train[s_tr].filter(pl.Series(mask_train_pre))
                meta_te = self.meta_train[s_te].filter(pl.Series(mask_test_pre))

                sc = 1.0
                try:
                    sc = strategy.score(
                        Xtr_tree,
                        Xtr_time,
                        ytr_tree,
                        Xte_tree,
                        Xte_time,
                        yte_tree,
                        treenames=self.treenames,
                        timenames=self.timenames,
                        meta_train=meta_tr,
                        meta_test=meta_te,
                        opt_params=opt_params,
                    )
                    if sc is not None and np.isfinite(sc):
                        n_valid = n_valid + 1
                        sc = float(sc)
                    else:
                        sc = 1.0
                except Exception as exc:
                    logger.exception("Score computation failed: %s", exc)
                    sc = 1.0

                scores.append(float(sc))

            logger.info("Scores per split: %s", scores)
            logger.info("Number of valid scores: %s", n_valid)
            logger.info("Geometric mean of scores: %s", float(np.exp(np.mean(np.log(np.array(scores))))))
            logger.info("Arithmetic mean of scores: %s", float(np.mean(np.array(scores))))
            logger.info("Variance of scores: %s", float(np.var(np.array(scores))))
            logger.info("Standard deviation of scores: %s", float(np.std(np.array(scores))))
            logger.info("Min score: %s", float(np.min(np.array(scores))))
            logger.info("Max score: %s", float(np.max(np.array(scores))))
            logger.info("Sharpe ratio (mean/std): %s", (float(np.mean(np.array(scores))) - 1.0) / np.std(np.array(scores)))

            if np.any(np.array(scores) <= 1e-4):
                logger.info("Pruning trial %s due to negative scores.", trial.number)
                logger.info("Num of negative scores: %s", np.sum(np.array(scores) <= 1e-4))
                raise optuna.TrialPruned()

            if n_valid < (len(slices) // 2):
                logger.info("Pruning trial %s due to insufficient valid scores.", trial.number)
                logger.info("Num of valid scores: %s", n_valid)
                logger.info("Num of splits: %s", len(slices))
                raise optuna.TrialPruned()
            
            scores_log = np.log(np.array(scores))
            return float(np.mean(scores_log))

        return objective

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_training_arrays(self, ls: LoadupSamples) -> None:
        """Extract training arrays and metadata from ``LoadupSamples``."""

        self.X_tree = ls.train_Xtree
        self.y_tree = ls.train_ytree
        self.X_time = ls.train_Xtime
        self.y_time = ls.train_ytime
        self.treenames = ls.featureTreeNames
        self.timenames = ls.featureTimeNames
        self.meta_train = ls.meta_pl_train

        if self.meta_train is None:
            raise ValueError("LoadupSamples must provide training metadata for Optuna splits.")
