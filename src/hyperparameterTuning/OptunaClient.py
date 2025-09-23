import optuna
import datetime
import random
import polars as pl
import numpy as np

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta
from src.predictionModule.LoadupSamples import LoadupSamples
from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.MachineModels import MachineModels

import logging
logger = logging.getLogger(__name__)

class OptunaClient:
    def __init__(self, ls:LoadupSamples, n_splits: int, n_test_days: int, n_training_days: int):
        self.__extract_arrays(ls)
        self.n_splits = n_splits
        self.n_test_days = n_test_days
        self.n_training_days = n_training_days

    def get_split_dates(
        self,
        final_split_date: datetime.date,
        start_train_date: datetime.date
    ):
        start = start_train_date + datetime.timedelta(days=self.n_training_days-1)
        end   = final_split_date - datetime.timedelta(days=self.n_test_days)
        eligible = [
            start + datetime.timedelta(days=i) 
                for i in range((end - start).days + 1) 
                    if (start + datetime.timedelta(days=i)).weekday() < 5
        ]
        split_dates = sorted(random.sample(eligible, self.n_splits))

        # Logging
        for date in split_dates:
            logger.info(f"  Split date: {date}")
        logger.info(f"  n_reruns: {self.n_splits}")
        logger.info(f"  max_training_days: {self.n_training_days}")
        logger.info(f"  n_testdays: {self.n_test_days}")
        logger.info(f"  start_train_date: {start_train_date}")
        logger.info(f"  final_split_date: {final_split_date}")

    def get_pivots(self):
        dates_tr = self.meta_train['date'].unique().sort()
        dates_tr_idx = dfta(self.meta_train, 'date').getNextLowerOrEqualIndices(dates_tr)

        N = len(dates_tr_idx)
        lo = self.n_training_days - 1                    # min pivot (last train index)
        hi = N - self.n_test_days - 2              # max pivot
        eligible = list(range(lo, hi + 1))
        assert len(eligible) >= self.n_splits, f"Too few eligible pivots ({len(eligible)}) for n_splits={self.n_splits}"
        pivots = sorted(random.sample(eligible, self.n_splits))

        for p in pivots:
            logger.info(f"  Pivot {p}: Date {dates_tr[p]}")

    def get_slices(self):
        pivots = self.get_pivots()

        dates_tr = self.meta_train['date'].unique().sort()
        dates_tr_idx = dfta(self.meta_train, 'date').getNextLowerOrEqualIndices(dates_tr)

        slices = [None] * self.n_splits
        for i in range(self.n_splits):
            p = pivots[i]

            tr_l_idx = dates_tr_idx[p - self.n_training_days + 1]
            tr_u_idx = dates_tr_idx[p + 1] - 1
            te_l_idx = dates_tr_idx[p + 1]
            te_u_idx = dates_tr_idx[p + self.n_test_days + 1] - 1

            s_tr = slice(tr_l_idx, tr_u_idx)
            s_te = slice(te_l_idx, te_u_idx)
            slices[i] = (s_tr, s_te)

        return slices
    
    def __extract_arrays(self, ls:LoadupSamples):
        self.Xtr_tree = ls.train_Xtree
        self.ytr_tree = ls.train_ytree
        self.Xte_tree = ls.test_Xtree
        self.yte_tree = ls.test_ytree

        self.Xtr_time = ls.train_Xtime
        self.ytr_time = ls.train_ytime
        self.Xte_time  = ls.test_Xtime
        self.yte_time  = ls.test_ytime

        self.treenames   = ls.featureTreeNames
        self.timenames   = ls.featureTimeNames
        self.meta_train  = ls.meta_pl_train
        self.meta_test   = ls.meta_pl_test
    
    def make_objective(self, 
            strategy: BaseStrategy,
        ) -> callable[[optuna.Trial], float]:
        slices = self.get_slices()

        def objective(trial: optuna.Trial) -> float:
            opt_params = strategy.sample_params(trial)
            logger.info(f"Trial {trial.number} with params: {opt_params}")

            scores = []
            mm = MachineModels(params=opt_params)
            for i in range(self.n_splits):
                s_tr, s_te = slices[i]

                score_params = {
                    "mm": mm,
                }
                sc = strategy.score(
                    self.Xtr_tree[s_tr],
                    self.Xtr_time[s_tr],
                    self.ytr_tree[s_tr],
                    self.Xte_tree[s_te],
                    self.Xte_time[s_te],
                    self.yte_tree[s_te],
                    params = score_params
                )

                #except Exception as e:
                #    logger.info(f"Exception during scoring: {e}")
                #    trial.should_prune()
                #    sc = metric(1.0)
                scores.append(sc)

            vals = [v for v in scores if np.isfinite(v)]
            logger.info(f"Scores per splits: {vals}")
            vals = np.array(vals)
            vals_log = np.log(1.0 + vals)
            if len(vals) < (len(scores)//2):
                return 0.0
            return float(np.mean(vals_log)) if len(vals_log) else -np.inf
        return objective