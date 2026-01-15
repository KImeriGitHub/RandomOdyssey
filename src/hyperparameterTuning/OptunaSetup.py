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
from src.hyperparameterTuning.HelperMetrics import HelperMetrics

logger = logging.getLogger(__name__)

SlicePair = Tuple[slice, slice]

class OptunaSetup:
    """Utility orchestrating Optuna optimisation on top of LoadupSamples."""

    def __init__(
        self,
        ls: LoadupSamples,
        n_splits: int,
        n_test_idxdays: int,
        n_training_idxdays: int,
        *,
        eval_mode: str = "all",
        spread_cost: float = 0.001,
        commission: float = 0.0000,
        min_targets_perday: int = 1,
        rng: Optional[random.Random] = None,
        model_cls: Optional[type] = None,
    ) -> None:
        self._extract_training_arrays(ls)
        self.n_splits = n_splits
        self.n_test_idxdays = n_test_idxdays
        self.n_training_idxdays = n_training_idxdays
        self._rng = rng if rng is not None else random.Random()
        if model_cls is None:
            model_cls = _MachineModels
        self._model_cls = model_cls
        self.eval_mode = eval_mode  
        self.spread_cost = spread_cost
        self.commission = commission
        self.min_targets_perday = min_targets_perday

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------
    def get_split_dates(
        self,
        final_split_date: datetime.date,
        start_train_date: datetime.date,
    ) -> List[datetime.date]:
        """Return random split dates respecting the provided constraints."""
        n_test_days = int(self.n_test_idxdays * 7 / 5)  # convert idx days to calendar days

        start = start_train_date + datetime.timedelta(int(days=self.n_training_idxdays*7/5) - 1)
        end = final_split_date - datetime.timedelta(days=n_test_days)
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
        logger.info("  max_training_days: %s", int(days=self.n_training_idxdays*7/5))
        logger.info("  n_testdays: %s", n_test_days)
        logger.info("  start_train_date: %s", start_train_date)
        logger.info("  final_split_date: %s", final_split_date)

        return split_dates

    def get_pivots(self) -> List[int]:
        """Sample pivot indices (last training day) used for slicing."""

        dates_tr = self.meta_train["date"].unique().sort()
        dates_tr_idx = dfta(self.meta_train, "date").getNextLowerOrEqualIndices(dates_tr)

        N = len(dates_tr_idx)
        lo = self.n_training_idxdays
        hi = (N - 1) - (self.n_test_idxdays)
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
            if p + self.n_test_idxdays + 1 > N:
                raise ValueError("Pivot too far to the end of training dates.")

            tr_l_idx = dates_tr_idx[p - self.n_training_idxdays + 1]
            tr_u_idx = dates_tr_idx[p + 1] - 1
            te_l_idx = dates_tr_idx[p + 1]
            if p + self.n_test_idxdays + 1 == N:
                te_u_idx = len(self.meta_train["date"]) - 1
            else:
                te_u_idx = dates_tr_idx[p + self.n_test_idxdays + 1] - 1

            s_tr = slice(tr_l_idx, tr_u_idx + 1)
            s_te = slice(te_l_idx, te_u_idx + 1)
            slices[i] = (s_tr, s_te)

        return slices

    # ------------------------------------------------------------------
    # Objective helper
    # ------------------------------------------------------------------
    def _log_scores_stats(self, scores: Sequence[float], roll_wndw: int) -> None:
        arr = np.array(scores, dtype=float)
        try:
            geo_mean = float(np.exp(np.mean(np.log(arr))))
        except Exception:
            geo_mean = float("nan")
        arith_mean = float(np.mean(arr))
        variance = float(np.var(arr))
        std_dev = float(np.std(arr))
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        sharpe = (arith_mean - 1.0) / std_dev if std_dev > 1e-6 else float("inf")

        roll = np.convolve(arr, np.ones(roll_wndw) / roll_wndw, mode="valid") if len(arr) >= roll_wndw else arr
        roll_std = float(np.std(roll)) if roll.size > 0 else float("nan")

        mid = len(arr) // 2
        mean_first_half = float(np.mean(arr[:mid])) if mid > 0 else float("nan")
        mean_second_half = float(np.mean(arr[mid:])) if len(arr) - mid > 0 else float("nan")
        mean_first_quarter = float(np.mean(arr[:len(arr)//4])) if len(arr)//4 > 0 else float("nan")
        mean_second_quarter = float(np.mean(arr[len(arr)//4:len(arr)//2])) if len(arr)//4 > 0 else float("nan")
        mean_third_quarter = float(np.mean(arr[len(arr)//2:3*len(arr)//4])) if len(arr)//4 > 0 else float("nan")
        mean_fourth_quarter = float(np.mean(arr[3*len(arr)//4:])) if len(arr)//4 > 0 else float("nan")

        # Logging
        logger.info("Geometric mean of scores: %s", geo_mean)
        logger.info("Arithmetic mean of scores: %s", arith_mean)
        logger.info("Variance of scores: %s", variance)
        logger.info("Standard deviation of scores: %s", std_dev)
        logger.info("Min score: %s", min_v)
        logger.info("Max score: %s", max_v)
        logger.info("Sharpe ratio (mean/std): %s", sharpe)
        logger.info("Std of rollingmean scores: %s", roll_std)
        logger.info(f"Mean first half {mean_first_half} vs second half {mean_second_half}")
        logger.info(f"Mean 1st quarter {mean_first_quarter} vs 2nd quarter {mean_second_quarter} vs 3rd quarter {mean_third_quarter} vs 4th quarter {mean_fourth_quarter}")

        return None

    def make_objective(self, strategy: BaseStrategy, preset_params: dict) -> Callable[[optuna.Trial], float]:
        """Create an Optuna objective callable using the provided strategy."""

        slices = self.get_slices()

        #######################
        ## Check loaded data ##
        #######################
        for key, val in strategy.expected_load_params.items():
            if key in preset_params.keys():
                if preset_params[key] != val:
                    raise ValueError(f"Preset param {key} has value {preset_params[key]}, expected {val}.")

        ####################
        ## PRE-PROCESSING ##
        ####################
        roll_wndw = max(2, self.n_splits // 10)
        preprocess_masks = [None] * self.n_splits
        scores = [None] * self.n_splits
        scores_direct = [None] * self.n_splits
        for i, (s_tr, s_te) in enumerate(slices):
            Xtr_tree = self.X_tree[s_tr]
            Xtr_time = self.X_time[s_tr]
            ytr_tree = self.y_tree[s_tr]
            ytr_tree_low  = self.y_tree_low[s_tr]
            ytr_tree_high = self.y_tree_high[s_tr]
            ytr_tree_open = self.y_tree_open[s_tr]
            Xte_tree = self.X_tree[s_te]
            Xte_time = self.X_time[s_te]
            yte_tree = self.y_tree[s_te]
            yte_tree_low  = self.y_tree_low[s_te]
            yte_tree_high = self.y_tree_high[s_te]
            yte_tree_open = self.y_tree_open[s_te]

            meta_train_slice = self.meta_train[s_tr] if self.meta_train is not None else None
            meta_test_slice = self.meta_train[s_te] if self.meta_train is not None else None

            def default_res():
                full_mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
                full_mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
                m_tr, m_te = full_mask_train, full_mask_test
                sl_tr = 0.86 * np.ones(ytr_tree.shape, dtype=float)
                sl_te = 0.86 * np.ones(yte_tree.shape, dtype=float)
                tp_tr = 2 * np.ones(ytr_tree.shape, dtype=float)
                tp_te = 2 * np.ones(yte_tree.shape, dtype=float)

                return m_tr, m_te, sl_tr, sl_te, tp_tr, tp_te

            try:
                mask_tr_pre, mask_te_pre, sl_tr, sl_te, tp_tr, tp_te = strategy.precompute(
                    Xtr_tree,
                    Xtr_time,
                    ytr_tree,
                    ytr_tree_low,
                    ytr_tree_high,
                    ytr_tree_open,
                    Xte_tree,
                    Xte_time,
                    treenames=self.treenames,
                    timenames=self.timenames,
                    meta_train=meta_train_slice,
                    meta_test=meta_test_slice,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("mask_precompute failed: %s", exc)
                mask_tr_pre, mask_te_pre, sl_tr, sl_te, tp_tr, tp_te = default_res()

            preprocess_masks[i] = (mask_tr_pre, mask_te_pre)
            
            last_day_mask = (meta_test_slice["date"] == meta_test_slice["date"].max()).fill_null(False)
            if self.eval_mode == "last_day":
                mask_te_pre = mask_te_pre & last_day_mask.to_numpy()

            res_vec = HelperMetrics.collapse_sl_tp(
                yte_tree, yte_tree_low, yte_tree_high, yte_tree_open, 
                sl_te, tp_te, 
                spread_cost=self.spread_cost, commission=self.commission
            )
            #sl_hits = (yte_tree_low[:, -1] <= sl_te)
            #tp_hits = (yte_tree_high[:, -1] >= tp_te)
            #tp_nosl_hits = tp_hits & ~sl_hits
            #logger.info(f"ratio sl hits test split {i}: {np.sum(sl_hits)/len(sl_te):.4f}, n_test_samples = {len(sl_te)}")
            #logger.info(f"ratio tp hits test split {i}: {np.sum(tp_hits)/len(tp_te):.4f}, n_test_samples = {len(tp_te)}")
            #logger.info(f"ratio tp no sl hits test split {i}: {np.sum(tp_nosl_hits)/len(tp_te):.4f}, n_test_samples = {len(tp_te)}")
            
            if self.eval_mode == "last_day":
                scores[i] = HelperMetrics.evaluate_mask_nullonempty(mask_te_pre, meta_test_slice['date'], res_vec)
                scores_direct[i] = HelperMetrics.evaluate_mask_nullonempty(mask_te_pre, meta_test_slice['date'], yte_tree[:, -1])
            else:
                scores[i] = HelperMetrics.evaluate_mask_mintarday(mask_te_pre, meta_test_slice['date'], res_vec, min_targets_perday=self.min_targets_perday)
                scores_direct[i] = HelperMetrics.evaluate_mask_mintarday(mask_te_pre, meta_test_slice['date'], yte_tree[:, -1], min_targets_perday=self.min_targets_perday)

        scores = [float(s) if s is not None and np.isfinite(s) else 1.0 for s in scores]
        scores_direct = [float(s) if s is not None and np.isfinite(s) else 1.0 for s in scores_direct]
        logger.info("Preprocessing complete.")
        logger.info("Precomputed scores per split (geometric mean of y_test with sl and tp): %s", scores)
        self._log_scores_stats(scores, roll_wndw)

        logger.info("")
        logger.info("Precomputed scores_direct per split (geometric mean of y_test): %s", scores_direct)
        self._log_scores_stats(scores_direct, roll_wndw)

        ######################
        ## Precompute indices
        ######################
        # for speed up during optimization, precompute indices for each  split
        fold_data = []
        for i, (s_tr, s_te) in enumerate(slices):
            mask_train_pre, mask_test_pre = preprocess_masks[i]

            # precompute integer indices once (much smaller than full arrays)
            idx_train = np.flatnonzero(mask_train_pre)
            idx_test  = np.flatnonzero(mask_test_pre)

            fold_data.append(
                dict(
                    s_tr=s_tr,
                    s_te=s_te,
                    idx_tr=idx_train,
                    idx_te=idx_test,
                )
            )

        ######################
        ## Objective function
        ######################
        def objective(trial: optuna.Trial) -> float:
            opt_params = strategy.sample_params(trial)
            logger.info("Trial %s with params: %s", trial.number, opt_params)

            scores = []
            scores_dir = []
            n_valid = 0

            for i, fold in enumerate(fold_data):
                s_tr   = fold["s_tr"]
                s_te   = fold["s_te"]
                idx_tr = fold["idx_tr"]
                idx_te = fold["idx_te"]

                # slicing and fancy indexing
                Xtr_tree      = self.X_tree[s_tr][idx_tr]
                ytr_tree      = self.y_tree[s_tr][idx_tr]
                ytr_tree_low  = self.y_tree_low[s_tr][idx_tr]
                ytr_tree_high = self.y_tree_high[s_tr][idx_tr]
                ytr_tree_open = self.y_tree_open[s_tr][idx_tr]

                Xte_tree      = self.X_tree[s_te][idx_te]
                yte_tree      = self.y_tree[s_te][idx_te]
                yte_tree_low  = self.y_tree_low[s_te][idx_te]
                yte_tree_high = self.y_tree_high[s_te][idx_te]
                yte_tree_open = self.y_tree_open[s_te][idx_te]

                Xtr_time      = self.X_time[s_tr][idx_tr]
                Xte_time      = self.X_time[s_te][idx_te]
                
                meta_train_sliced = self.meta_train[s_tr]
                meta_test_sliced  = self.meta_train[s_te]
                meta_tr       = meta_train_sliced[idx_tr]
                meta_te       = meta_test_sliced[idx_te]

                sc = 1.0
                sc_dir = 1.0
                try:
                    _, mask_te_run, _, sl_te, _, tp_te, _, _ = strategy.run(
                        Xtr_tree,
                        Xtr_time,
                        ytr_tree,
                        ytr_tree_low,
                        ytr_tree_high,
                        ytr_tree_open,
                        Xte_tree,
                        Xte_time,
                        treenames=self.treenames,
                        timenames=self.timenames,
                        meta_train=meta_tr,
                        meta_test=meta_te,
                        opt_params=opt_params,
                    )
                    
                    res_vec = HelperMetrics.collapse_sl_tp(
                        yte_tree, 
                        yte_tree_low, 
                        yte_tree_high, 
                        yte_tree_open, 
                        sl_te, tp_te, 
                        spread_cost=self.spread_cost, 
                        commission=self.commission
                    )
                    
                    mask_full_te_run = np.zeros(len(meta_test_sliced), dtype=bool)
                    mask_full_te_run[idx_te] = mask_te_run
                    
                    res_vec_full = np.zeros(len(meta_test_sliced), dtype=float)
                    res_vec_full[idx_te] = res_vec
                    
                    if self.eval_mode == "last_day":
                        last_day_mask = (meta_test_sliced["date"] == meta_test_sliced["date"].max()).fill_null(False).to_numpy()
                        mask_full_te_run = mask_full_te_run & last_day_mask
                        
                        sc = HelperMetrics.evaluate_mask_nullonempty(mask_full_te_run, meta_test_sliced['date'], res_vec_full)
                        sc_dir = HelperMetrics.evaluate_mask_nullonempty(mask_full_te_run, meta_test_sliced['date'], self.y_tree[s_te][:, -1])
                    else:
                        sc = HelperMetrics.evaluate_mask_mintarday(mask_full_te_run, meta_test_sliced['date'], res_vec_full, min_targets_perday=self.min_targets_perday)
                        sc_dir = HelperMetrics.evaluate_mask_mintarday(mask_full_te_run, meta_test_sliced['date'], self.y_tree[s_te][:, -1], min_targets_perday=self.min_targets_perday)
                        
                    if sc is not None and np.isfinite(sc):
                        n_valid = n_valid + 1
                        sc = float(sc)
                        sc_dir = float(sc_dir)
                    else:
                        sc = 1.0
                        sc_dir = 1.0

                    logger.info(f"Split {i}: Score = {sc}, Score (no SL/TP) = {sc_dir}")
                    logger.info(f"  median SL = {float(np.median(sl_te))}, median TP = {float(np.median(tp_te))}, max SL = {float(np.max(sl_te))}, max TP = {float(np.max(tp_te))}, min SL = {float(np.min(sl_te))}, min TP = {float(np.min(tp_te))}")
                    logger.info(f"  ratio test samples = {mask_te_run.sum()/len(mask_te_run):.4f}, n_test_samples = {mask_te_run.sum()}")

                except Exception as exc:
                    logger.exception("Score computation failed: %s", exc)
                    sc = 1.0
                    sc_dir = 1.0

                scores.append(float(sc))
                scores_dir.append(float(sc_dir))

            logger.info("Scores per split: %s", scores)
            logger.info("Number of valid scores: %s", n_valid)
            self._log_scores_stats(scores, roll_wndw)

            logger.info("")
            logger.info("Scores no sl and tp per split: %s", scores_dir)
            self._log_scores_stats(scores_dir, roll_wndw)

            if np.any(np.array(scores) <= 1e-6):
                logger.info("Pruning trial %s due to negative scores.", trial.number)
                logger.info("Num of negative scores: %s", np.sum(np.array(scores) <= 1e-4))
                raise optuna.TrialPruned()

            if n_valid < (len(slices) // 2):
                logger.info("Pruning trial %s due to insufficient valid scores.", trial.number)
                logger.info("Num of valid scores: %s", n_valid)
                logger.info("Num of splits: %s", len(slices))
                raise optuna.TrialPruned()
            
            scores_log = np.log(np.array(scores))
            
            #n = max(len(scores_log) // 4, 5)
            #p = np.linspace(0, 1, n)                   # linspace 0 to 1
            #q = np.quantile(scores_log, p)             # q_i at p
            #w = 1 - p**0                               # w_i
            #w = w / w.sum()                            # normalization
            #
            #score_fin = float(np.exp(np.sum(w * q)))
            
            return np.exp(np.mean(scores_log))

        return objective

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_training_arrays(self, ls: LoadupSamples) -> None:
        """Extract training arrays and metadata from ``LoadupSamples``."""

        self.X_tree = ls.train_Xtree
        self.y_tree = ls.train_ytree
        self.y_tree_low = ls.train_ytree_low
        self.y_tree_high = ls.train_ytree_high
        self.y_tree_open = ls.train_ytree_open
        self.X_time = ls.train_Xtime
        self.y_time = ls.train_ytime
        self.treenames = ls.featureTreeNames
        self.timenames = ls.featureTimeNames
        self.meta_train = ls.meta_pl_train

        if self.meta_train is None:
            raise ValueError("LoadupSamples must provide training metadata for Optuna splits.")

        # Make arrays read-only
        for name in [
            "X_tree",
            "y_tree",
            "y_tree_low",
            "y_tree_high",
            "y_tree_open",
            "X_time",
            "y_time",
        ]:
            arr = getattr(self, name, None)
            if arr is not None and hasattr(arr, "setflags"):
                arr.setflags(write=False)
