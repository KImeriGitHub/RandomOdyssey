import os
import numpy as np
import pandas as pd
import polars as pl
import datetime
import re
import copy
import copy as _copy
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

import pandas_market_calendars as mcal

import logging
logger = logging.getLogger(__name__)

class LoadupSamples:
    DEFAULT_PARAMS = {
        "daysAfterPrediction": None,
        "idxAfterPrediction": 10,
        'timesteps': 20,
        'target_option': 'last',
        
        "LoadupSamples_time_inc_factor": 10,
        "LoadupSamples_tree_scaling_standard": True,
        "LoadupSamples_time_scaling_stretch": True,
        
    }
    def __init__(self, 
            train_start_date: datetime.date,
            test_dates: list[datetime.date],
            treegroup: str | None = None,
            timegroup: str | None = None,
            params: dict = None,
        ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.train_start_date = train_start_date
        self.test_dates = test_dates
        self.treegroup = treegroup
        self.timegroup = timegroup

        if treegroup is None and timegroup is None:
            raise ValueError("Either treegroup or timegroup must be specified.")

        self.min_test_date = min(self.test_dates)
        self.max_test_date = max(self.test_dates)
        
        self.daysAfter = self.params.get('daysAfterPrediction', None)
        self.idxAfter = self.params.get('idxAfterPrediction', None)
        self.timesteps = self.params['timesteps']
        self.target_option = self.params['target_option']
        
        self.featureTreeNames: list[str] | None = None
        self.featureTimeNames: list[str] | None = None
        self.meta_pl_train: pl.DataFrame | None = None
        self.meta_pl_test: pl.DataFrame | None = None
        self.test_Xtree: np.array | None = None
        self.test_Xtime: np.array | None = None
        self.test_ytree: np.array | None = None
        self.test_ytime: np.array | None = None
        self.train_Xtree: np.array | None = None
        self.train_Xtime: np.array | None = None
        self.train_ytree: np.array | None = None
        self.train_ytime: np.array | None = None

    def dataset_tests(self) -> None:
        """
        Perform simple tests on the training and test datasets.
        """
        is_test_env = (
            self.test_ytree is not None 
            and self.test_ytime is not None
        )
        
        # Check for same shape
        if self.treegroup is not None:
            if self.train_Xtree.shape[1] != self.test_Xtree.shape[1]:
                logger.error("Training and test datasets have different number of features.")
            if self.train_ytree.shape[0] != self.train_Xtree.shape[0]:
                logger.error("Training labels do not match the number of training samples.")
            if is_test_env:
                if self.test_ytree.shape[0] != self.test_Xtree.shape[0]:
                    logger.error("Test labels do not match the number of test samples.")
            if self.train_Xtree.shape[1] != len(self.featureTreeNames):
                logger.error("Number of features in training data does not match the number of tree feature names.")
            if len(self.meta_pl_train['date']) != self.train_Xtree.shape[0]:
                logger.error("Number of sample dates does not match the number of training samples.")

        if self.timegroup is not None:
            if self.train_Xtime.shape[2] != self.test_Xtime.shape[2]:
                logger.error("Training and test datasets have different number of features.")
            if self.train_ytime.shape[0] != self.train_Xtime.shape[0]:
                logger.error("Training labels do not match the number of training samples.")
            if is_test_env:
                if self.test_ytime.shape[0] != self.test_Xtime.shape[0]:
                    logger.error("Test labels do not match the number of test samples.")
            if self.train_Xtime.shape[2] != len(self.featureTimeNames):
                logger.error("Number of features in training data does not match the number of time feature names.")
            if len(self.meta_pl_train['date']) != self.train_Xtime.shape[0]:
                logger.error("Number of sample dates does not match the number of training samples.")
        
        if not self.meta_pl_train['date'].is_sorted():
            logger.error("Sample dates are not sorted. Please sort them before proceeding.")
        if not self.meta_pl_test['date'].is_sorted():
            logger.error("Test Sample dates are not sorted. Please sort them before proceeding.")
        if not all(isinstance(date, datetime.date) for date in self.meta_pl_train['date']):
            logger.error("Sample dates should be pandas Timestamp objects.")
        if is_test_env and not all(isinstance(date, datetime.date) for date in self.meta_pl_test['date']):
            logger.error("Sample dates for test set should be pandas Timestamp objects.")

    def copy(self, *, deep: bool = True) -> "LoadupSamples":
        """Return a copy of this instance.
        deep=True clones numpy arrays, polars DataFrames and params; deep=False shares them.
        """

        # Recreate base object
        new = LoadupSamples(
            train_start_date=self.train_start_date,
            test_dates=(list(self.test_dates) if deep else self.test_dates),
            treegroup=self.treegroup,
            timegroup=self.timegroup,
            params=(_copy.deepcopy(self.params) if deep else dict(self.params)),
        )

        # Preserve (possibly post-init) derived/overridden fields
        new.min_test_date = self.min_test_date
        new.max_test_date = self.max_test_date
        new.daysAfter = self.daysAfter
        new.idxAfter = self.idxAfter
        new.timesteps = self.timesteps
        new.target_option = self.target_option

        # Helpers
        def _c_arr(a): return None if a is None else (a.copy() if deep else a)
        def _c_list(v): return None if v is None else (list(v) if deep else v)
        def _c_pl(df): return None if df is None else (df.clone() if deep else df)

        # Metadata
        new.featureTreeNames = _c_list(self.featureTreeNames)
        new.featureTimeNames = _c_list(self.featureTimeNames)
        new.meta_pl_train = _c_pl(self.meta_pl_train)
        new.meta_pl_test = _c_pl(self.meta_pl_test)

        # Train/Test arrays
        new.train_Xtree = _c_arr(self.train_Xtree)
        new.train_Xtime = _c_arr(self.train_Xtime)
        new.train_ytree = _c_arr(self.train_ytree)
        new.train_ytime = _c_arr(self.train_ytime)
        new.test_Xtree  = _c_arr(self.test_Xtree)
        new.test_Xtime  = _c_arr(self.test_Xtime)
        new.test_ytree  = _c_arr(self.test_ytree)
        new.test_ytime  = _c_arr(self.test_ytime)

        return new

    def load_samples(self, main_path: str = "src/featureAlchemy/bin/") -> Optional[pl.DataFrame]:
        years = np.arange(self.train_start_date.year, self.max_test_date.year + 1)
        
        names_treedata: list = []
        names_timedata: list = []
        meta_alltree_pl_list = []
        meta_alltime_pl_list = []
        alltree_X_list = []
        alltime_X_list = []

        prefix_tree = f"TreeFeatures"
        prefix_time = f"TimeFeatures"

        # check whether files are there otherwise make error
        for year in years:
            label = str(year)
            if self.treegroup is not None and not os.path.isfile(main_path + f"{prefix_tree}_{label}_{self.treegroup}.npz"):
                raise FileNotFoundError(f"{prefix_tree} sets not found.")
            if self.timegroup is not None and not os.path.isfile(main_path + f"{prefix_time}_{label}_{self.timegroup}.npz"):
                raise FileNotFoundError(f"{prefix_time} sets not found.")

        # Load data
        for year in years:
            for cat in ["Tree", "Time"]:
                if cat == "Tree" and self.treegroup is None:
                    continue
                if cat == "Time" and self.timegroup is None:
                    continue

                label = str(year)

                if cat == "Tree":
                    path_npz = main_path + f"{cat}Features_{label}_{self.treegroup}.npz"
                if cat == "Time":
                    path_npz = main_path + f"{cat}Features_{label}_{self.timegroup}.npz"

                data = np.load(path_npz, allow_pickle=True)

                meta_data  = data['meta_feat']
                feats_data = data['featuresArr']
                names_data = data['featuresNames']
            
                # shorten to timesteps
                if cat == "Time" and feats_data.shape[1] > self.timesteps:
                    feats_data = feats_data[:,-self.timesteps:,:]

                # meta polars
                meta_pl_loop: pl.DataFrame = pl.DataFrame({
                    "date":     meta_data["date"],
                    "ticker":   meta_data["ticker"],
                    "Close":    meta_data["Close"],
                    "AdjClose": meta_data["AdjClose"],
                    "Open":     meta_data["Open"],
                })

                if cat == "Tree":
                    names_treedata = names_data
                    meta_alltree_pl_list.append(meta_pl_loop)
                    alltree_X_list.append(feats_data)
                if cat == "Time":
                    names_timedata = names_data
                    meta_alltime_pl_list.append(meta_pl_loop)
                    alltime_X_list.append(feats_data)

        # Post processing
        if self.treegroup is not None:
            metatree_pl: pl.DataFrame = pl.concat(meta_alltree_pl_list)
            alltree_X_pre: np.ndarray = np.concatenate(alltree_X_list, axis=0)
        if self.timegroup is not None:
            metatime_pl: pl.DataFrame = pl.concat(meta_alltime_pl_list)
            alltime_X_pre: np.ndarray = np.concatenate(alltime_X_list, axis=0)

        if self.treegroup is not None and self.timegroup is not None:
            meta_pl, all_Xtree_pre, all_Xtime_pre = self.__combine_tree_and_time(
                metatree_pl, metatime_pl, alltree_X_pre, alltime_X_pre
            )
        if self.treegroup is None and self.timegroup is not None:
            meta_pl = metatime_pl
            all_Xtree_pre = None
            all_Xtime_pre = alltime_X_pre
        if self.treegroup is not None and self.timegroup is None:
            meta_pl = metatree_pl
            all_Xtree_pre = alltree_X_pre
            all_Xtime_pre = None

        # Check if test dates are trading dates and modify if necessary
        modified_test_dates = copy.deepcopy(self.test_dates)
        for idx, test_date in enumerate(modified_test_dates):
            if not meta_pl["date"].is_in(pl.Series([test_date])).any():
                logger.warning(f"Test date {test_date} not found in the database. Omitting.")
                modified_test_dates[idx] = meta_pl.filter(pl.col("date") <= test_date).select("date").max().item()
        self.test_dates = sorted(set(modified_test_dates))
        self.min_test_date = self.test_dates[0]
        self.max_test_date = self.test_dates[-1]

        # Add target
        meta_pl = meta_pl.sort(["date", "ticker"])
        meta_pl = self.__add_target(meta_pl)
        meta_pl = meta_pl.sort(["date", "ticker"])
        
        # Assign Main Variables
        mask_at_test_dates = (
            (meta_pl["date"] >= self.min_test_date)
            & (meta_pl["date"] <= self.max_test_date)
        ).fill_null(False).to_numpy()
        mask_inbetween_date = (
            (meta_pl["date"] >= self.train_start_date)
            & (meta_pl["date"] < self.min_test_date)
        ).fill_null(False).to_numpy()
        
        self.meta_pl_train = meta_pl.filter(pl.Series(mask_inbetween_date))
        self.meta_pl_test  = meta_pl.filter(pl.Series(mask_at_test_dates))
        
        if self.treegroup is not None:
            self.featureTreeNames = list(names_treedata)
            self.train_Xtree = all_Xtree_pre[mask_inbetween_date]
            self.test_Xtree = all_Xtree_pre[mask_at_test_dates]
            
        if self.timegroup is not None:
            self.featureTimeNames = list(names_timedata)
            self.train_Xtime = all_Xtime_pre[mask_inbetween_date]
            self.test_Xtime = all_Xtime_pre[mask_at_test_dates]
        
        # Assign target
        tar_all = meta_pl["target_ratio"].to_numpy().flatten()
        
        rat_inbetween = tar_all[mask_inbetween_date]
        rat_at_test_date = tar_all[mask_at_test_dates]
        
        if self.treegroup is not None:
            self.train_ytree = np.clip(rat_inbetween, 1e-2, 1e2)
            self.test_ytree = np.clip(rat_at_test_date, 1e-2, 1e2)
        
        if self.timegroup is not None:
            inc_factor = self.params["LoadupSamples_time_inc_factor"]
            self.train_ytime = np.tanh((rat_inbetween - 1.0) * inc_factor) / 2.0 + 0.5
            self.test_ytime = np.tanh((rat_at_test_date - 1.0) * inc_factor) / 2.0 + 0.5
        
        # Clean Data
        self.__remove_nan_samples_train()
        
        # Scaling Tree
        if self.treegroup is not None and self.params["LoadupSamples_tree_scaling_standard"]:
            self.__scale_tree_standard()
        
        # Scaling Time
        if self.timegroup is not None and self.params["LoadupSamples_time_scaling_stretch"]:
            self.train_Xtime = self.__scale_time_stretch(self.train_Xtime)
            self.test_Xtime = self.__scale_time_stretch(self.test_Xtime)
            
        # Final checks
        self.dataset_tests()

    def __combine_tree_and_time(self, 
            meta_tree: pl.DataFrame, 
            meta_time: pl.DataFrame, 
            all_tree_X: np.ndarray, 
            all_time_X: np.ndarray
        ) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        metatree_idx = meta_tree.with_row_index("idx_tree")
        metatime_idx = meta_time.with_row_index("idx_time")
        meta_pl = metatree_idx.join(metatime_idx, on=["date", "ticker"], how="inner", suffix="_time")
        idx_tree = meta_pl["idx_tree"].to_numpy()
        idx_time = meta_pl["idx_time"].to_numpy()

        alltree_X = all_tree_X[idx_tree]
        alltime_X = all_time_X[idx_time]

        # Assert resulting np length
        assert len(idx_tree) == len(idx_time), "Tree and Time indices must match in length."

        # Assert Close, Open etc match in combined dataframe
        cols = ["Close", "Open"] #Adj Close might differ, that is why it is excluded
        left = meta_pl.select(cols)
        right = meta_pl.select([f"{c}_time" for c in cols]).rename(
            {f"{c}_time": c for c in cols}
        )
        assert left.equals(right, null_equal=True), "Mismatch between cols and their *_time counterparts."
        
        meta_pl = meta_pl.drop(["idx_tree", "idx_time", "AdjClose_time","Close_time","Open_time"])

        return meta_pl, alltree_X, alltime_X

    def split_dataset(self, 
        start_date: datetime.date, 
        last_train_date: datetime.date, 
        last_test_date: datetime.date = None
    ) -> None:
        if last_train_date >= self.meta_pl_test.select(pl.min('date')).item():
            raise ValueError(
                f"Last training date {last_train_date} must be before the first test date {self.meta_pl_test.select(pl.min('date')).item()}"
            )
            
        if last_test_date is None:
            last_test_date = self.meta_pl_test.select(pl.max('date')).item()
        
        # Create a mask for training data
        train_split_tr_mask = (
            (self.meta_pl_train["date"] >= start_date) & 
            (self.meta_pl_train["date"] <= last_train_date)
        ).fill_null(False).to_numpy()
        train_split_te_mask = (
            (self.meta_pl_train["date"] > last_train_date) & 
            (self.meta_pl_train["date"] <= last_test_date)
        ).fill_null(False).to_numpy()
        test_split_mask = (
            (self.meta_pl_test["date"] > last_train_date) & 
            (self.meta_pl_test["date"] <= last_test_date)
        ).fill_null(False).to_numpy()

        # Tree
        if self.treegroup is not None:
            self.test_Xtree = np.concatenate(
                [
                    self.train_Xtree[train_split_te_mask], 
                    self.test_Xtree[test_split_mask]
                ], 
                axis = 0
            )
            self.test_ytree = np.concatenate(
                [
                    self.train_ytree[train_split_te_mask], 
                    self.test_ytree[test_split_mask]
                ], 
                axis = 0
            )

            self.train_Xtree = self.train_Xtree[train_split_tr_mask]
            self.train_ytree = self.train_ytree[train_split_tr_mask]

        # Time
        if self.timegroup is not None:
            self.test_Xtime = np.concatenate(
                [
                    self.train_Xtime[train_split_te_mask], 
                    self.test_Xtime[test_split_mask]
                ], 
                axis = 0
            )
            self.test_ytime = np.concatenate(
                [
                    self.train_ytime[train_split_te_mask], 
                    self.test_ytime[test_split_mask]
                ], 
                axis = 0
            )
            
            self.train_Xtime = self.train_Xtime[train_split_tr_mask]
            self.train_ytime = self.train_ytime[train_split_tr_mask]

        self.meta_pl_test = pl.concat([
            self.meta_pl_train.filter(pl.Series(train_split_te_mask)),
            self.meta_pl_test.filter(pl.Series(test_split_mask))
        ])
        self.meta_pl_train = self.meta_pl_train.filter(pl.Series(train_split_tr_mask))

        self.min_test_date = self.meta_pl_test.select(pl.min('date')).item()
        self.test_dates = self.meta_pl_test.select(pl.col('date')).to_series().unique().to_list()
        
        self.dataset_tests()

    def apply_masks(self, mask_train: np.ndarray, mask_test: np.ndarray | None) -> None:
        if mask_test is None:
            mask_test = np.ones((self.test_Xtree.shape[0],), dtype=bool)
        if self.treegroup is not None:
            self.train_Xtree = self.train_Xtree[mask_train]
            self.train_ytree = self.train_ytree[mask_train]
            self.test_Xtree = self.test_Xtree[mask_test]
            self.test_ytree = self.test_ytree[mask_test]
        
        if self.timegroup is not None:
            self.train_Xtime = self.train_Xtime[mask_train]
            self.train_ytime = self.train_ytime[mask_train]
            self.test_Xtime = self.test_Xtime[mask_test]
            self.test_ytime = self.test_ytime[mask_test]
        
        self.meta_pl_train = self.meta_pl_train.filter(pl.Series(mask_train))
        self.meta_pl_test = self.meta_pl_test.filter(pl.Series(mask_test))
        
        self.dataset_tests()
        
    def __add_target(self, meta_pl: pl.DataFrame):
        idx_after = self.__determine_idx_after(meta_pl)
        
        meta_pl = meta_pl.sort(["date", "ticker"])

        date_expr = pl.col("date").shift(-idx_after).over("ticker").alias("target_date")

        # get target close price after idx_after days
        last_expr = (
            pl.col("AdjClose")
            .shift(-idx_after)
            .over("ticker")
            .alias("target_last_close")
        )

        # get mean over all future close prices after idx_after days
        mean_expr = (
            pl.col("AdjClose")
            .shift(-idx_after)
            .rolling_mean(window_size=max(idx_after//2,1))
            .over("ticker")
            .alias("target_mean_close")
        )

        # get max over all future close prices after idx_after days
        max_expr = (
            pl.col("AdjClose")
            .shift(-idx_after)
            .rolling_max(window_size=idx_after)
            .over("ticker")
            .alias("target_max_close")
        )
        
        if self.target_option == 'last':
            option_expr = last_expr.alias("target_price")
        elif self.target_option == 'mean':
            option_expr = mean_expr.alias("target_price")
        elif self.target_option == 'max':
            option_expr = max_expr.alias("target_price")
        else:
            raise ValueError(f"Unknown target option: {self.target_option}. Choose from 'last', 'mean', or 'max'.")
        
        # For keeping track of intermediate prices
        allclose_exprs = [
            (
                pl.col("AdjClose").shift(-i).over("ticker")
                * (pl.col("Close") / pl.col("AdjClose"))
            ).alias(f"target_close_at{i}")
            for i in range(1, idx_after + 1)
        ]

        meta_pl = meta_pl.with_columns(
            [date_expr] + 
            allclose_exprs + 
            [last_expr] + [mean_expr] + [max_expr] + 
            [option_expr]
        )
        
        meta_pl = meta_pl.with_columns((pl.col("Open")*(pl.col("AdjClose")/pl.col("Close"))).alias("AdjOpen"))
        meta_pl = meta_pl.with_columns(pl.col("AdjOpen").shift(-1).over("ticker").alias("NextDayAdjOpen"))
        meta_pl = meta_pl.with_columns((pl.col("target_price") / pl.col("NextDayAdjOpen")).alias("target_ratio"))

        return meta_pl
    
    def __determine_idx_after(self, meta_pl: pl.DataFrame) -> int:
        """
        Determine the index after prediction based on either idxAfter or daysAfter.
        If both are provided, prefer idxAfter (with a warning if they disagree).
        """
        if self.idxAfter is None and self.daysAfter is None:
            raise ValueError("Either idxAfter or daysAfter must be specified.")
        
        # Compute based-on-days only if daysAfter was given
        if self.daysAfter is not None:
            calc_idx_after = self.__calc_trading_days(meta_pl, self.daysAfter)
        
        # If both are specified, compare and warn if they differ, then return idxAfter
        if self.idxAfter is not None and self.daysAfter is not None:
            if calc_idx_after != self.idxAfter:
                logger.warning(
                    "LoadupSamples: daysAfter (%d) -> calc_idx_after (%d) differs from "
                    "idxAfter (%d); defaulting to idxAfter.",
                    self.daysAfter, calc_idx_after, self.idxAfter
                )
            return self.idxAfter
        
        # If only idxAfter is specified
        if self.idxAfter is not None:
            return self.idxAfter
        
        # Otherwise, only daysAfter is specified
        return calc_idx_after
    
    def __calc_trading_days(self, meta_pl: pl.DataFrame, days_After: int) -> int:
        """
        Count NYSE trading days (excl. weekends & market holidays)
        between the last date in meta_pl and last_date + days_After.
        """
        # 1. extract last date
        last_date: datetime.date = (
            meta_pl
            .select(pl.col("date").max())
            .to_series()
            .item()
        )

        # 2. compute target date and start
        end_date = last_date + datetime.timedelta(days=days_After)
        start = last_date + datetime.timedelta(days=1)

        # 3. get NYSE schedule and count days
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=start, end_date=end_date)
        return sched.shape[0]
    
    def __remove_nan_samples_train(self) -> None:
        threshold_biggness = 1e10

        # TRAINING
        mask_train = np.ones(self.meta_pl_train.height, dtype=bool)
        if self.treegroup is not None:
            X = self.train_Xtree
            y = self.train_ytree
            bad_train_X = (~np.isfinite(X) | (np.abs(X) > threshold_biggness)).any(axis=1)
            bad_train_y = ~np.isfinite(y) | (np.abs(y) > threshold_biggness)
            if bad_train_y.sum():
                logger.info(f"Non-finite/too-large values in train y tree: {bad_train_y.sum()} samples.")
            if bad_train_X.sum():
                logger.info(f"Non-finite/too-large values in train X tree: {bad_train_X.sum()} samples.")
            mask_train &= ~(bad_train_X | bad_train_y)

        if self.timegroup is not None:
            bad_ytime = ~np.isfinite(self.train_ytime)
            if bad_ytime.sum() > 0:
                logger.info(f"Non-finite/too-large values in train y time: {bad_ytime.sum()} samples.")
            mask_train &= ~bad_ytime

        # Removing train samples
        if (~mask_train).sum():
            logger.info(f"Removing {(~mask_train).sum()} samples from training data.")
        if self.treegroup is not None:
            self.train_Xtree = self.train_Xtree[mask_train]
            self.train_ytree = self.train_ytree[mask_train]
        if self.timegroup is not None:
            self.train_Xtime = self.train_Xtime[mask_train]
            self.train_ytime = self.train_ytime[mask_train]
        self.meta_pl_train = self.meta_pl_train.filter(pl.Series(mask_train))

        # TESTING
        mask_test = np.ones(self.meta_pl_test.height, dtype=bool)
        if self.treegroup is not None:
            X = self.test_Xtree
            bad_test_X = (~np.isfinite(X) | (np.abs(X) > threshold_biggness)).any(axis=1)
            if bad_test_X.sum():
                logger.info(f"Non-finite/too-large values in test X tree: {bad_test_X.sum()} samples.")
            mask_test &= ~(bad_test_X)

        # Removing test samples
        if (~mask_test).sum():
            logger.info(f"Removing {(~mask_test).sum()} samples from testing data.")
        if self.treegroup is not None:
            self.test_Xtree = self.test_Xtree[mask_test]
            self.test_ytree = self.test_ytree[mask_test]
        if self.timegroup is not None:
            self.test_Xtime = self.test_Xtime[mask_test]
            self.test_ytime = self.test_ytime[mask_test]
        self.meta_pl_test = self.meta_pl_test.filter(pl.Series(mask_test))

    def __scale_tree_standard(self) -> None:
        scaler = StandardScaler()
        scaler.fit(self.train_Xtree)
        self.train_Xtree = scaler.transform(self.train_Xtree)
        self.test_Xtree  = scaler.transform(self.test_Xtree)
        
    def __scale_time_stretch(self, X: np.array) -> np.array:
        """
        Normalize each sample in time so that the midpoint 0.5 remains fixed, 
        stretching values around to hit either 0 or 1.
        """
        assert np.issubdtype(X.dtype, np.floating), "Input X must be a float array"
        
        center = 0.5

        # per‐sample min/max along time
        mins = X.min(axis=1, keepdims=True)
        maxs = X.max(axis=1, keepdims=True)

        # distances from center
        den_above = maxs - center
        den_below = center - mins
        den = np.maximum(den_above, den_below)

        # avoid div-by-zero
        eps = 1e-6
        den = np.where(np.abs(den)<eps, 1.0, den)

        # linear stretch 
        return np.clip((X - center) / den + center, 0.0, 1.0)