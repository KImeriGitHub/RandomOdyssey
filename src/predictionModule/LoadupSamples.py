import os
import numpy as np
import pandas as pd
import polars as pl
import datetime
import re
from typing import Optional
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
            group: str,
            group_type: str = 'Tree',
            params: dict = None,
        ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.group = group
        self.group_type = group_type
        self.train_start_date = train_start_date
        self.test_dates = test_dates
        
        self.min_test_date = min(self.test_dates)
        self.max_test_date = max(self.test_dates)
        
        # Assign parameters to instance variables
        
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
        if not self.meta_pl_train['date'].is_sorted():
            logger.error("Sample dates are not sorted. Please sort them before proceeding.")
        if not all(isinstance(date, datetime.date) for date in self.meta_pl_train['date']):
            logger.error("Sample dates should be pandas Timestamp objects.")
        if is_test_env and not all(isinstance(date, datetime.date) for date in self.meta_pl_test['date']):
            logger.error("Sample dates for test set should be pandas Timestamp objects.")

    def load_samples(self, main_path: str = "src/featureAlchemy/bin/") -> Optional[pl.DataFrame]:
        years = np.arange(self.train_start_date.year, self.max_test_date.year + 1)
        
        meta_all_pl_list = []
        names_data: list = []
        all_X_list = []
        
        prefix = f"{self.group_type}Features"
        
        # check whether files are there otherwise make error
        for year in years:
            label = str(year)
            if not os.path.isfile(main_path + f"{prefix}_{label}_{self.group}.npz"):
                raise FileNotFoundError(f"Feature sets not found.")
            
        # Load data
        testdate_in_db = np.zeros(len(self.test_dates), dtype=bool)
        for year in years:
            label = str(year)
            
            path_npz = main_path + f"{prefix}_{label}_{self.group}.npz"
            
            data = np.load(path_npz, allow_pickle=True)
            
            meta_data  = data['meta_feat']
            feats_data = data['featuresArr']
            names_data = data['featuresNames']
            
            for idx, test_date in enumerate(self.test_dates):
                if test_date in meta_data["date"]:
                    testdate_in_db[idx] = True
            
            # shorten to timesteps
            if self.group_type == "time" and feats_data.shape[1] > self.timesteps:
                feats_data = feats_data[:,-self.timesteps:,:]

            # meta polars
            meta_pl_all_loop: pl.DataFrame = pl.DataFrame({
                "date":     meta_data["date"],
                "ticker":   meta_data["ticker"],
                "Close":    meta_data["Close"],
                "AdjClose": meta_data["AdjClose"],
                "Open":     meta_data["Open"],
            })
            
            meta_all_pl_list.append(meta_pl_all_loop)
            all_X_list.append(feats_data)

        # Post processing
        meta_pl: pl.DataFrame = pl.concat(meta_all_pl_list)
        all_X_pre = np.concatenate(all_X_list, axis=0)
                
        # Check if test dates are trading dates and modify if necessary
        modified_test_dates = self.test_dates[:]
        for idx, test_date in enumerate(modified_test_dates):
            if not testdate_in_db[idx]:
                logger.warning(f"Test date {test_date} not found in the database. Resetting to last trading day.")
                modified_test_dates[idx] = meta_pl.filter(pl.col("date") <= test_date).select("date").max()["date"].item()
        self.test_dates = modified_test_dates

        # Add target
        meta_pl = meta_pl.sort(["date", "ticker"])
        meta_pl = self.__add_target(meta_pl)
        meta_pl = meta_pl.sort(["date", "ticker"])
        
        # Assign Main Variables
        mask_at_test_dates = (meta_pl["date"].is_in(self.test_dates)).fill_null(False).to_numpy()
        mask_inbetween_date = (
            (meta_pl["date"] >= self.train_start_date)
            & (meta_pl["target_date"] < self.min_test_date)
        ).fill_null(False).to_numpy()
        
        self.meta_pl_train = meta_pl.filter(mask_inbetween_date)
        self.meta_pl_test  = meta_pl.filter(mask_at_test_dates)
        
        if self.group_type == "Tree":
            self.featureTreeNames = names_data
            self.train_Xtree = all_X_pre[mask_inbetween_date]
            self.test_Xtree = all_X_pre[mask_at_test_dates]
            
        if self.group_type == "Time":
            self.featureTimeNames = names_data
            self.train_Xtime = all_X_pre[mask_inbetween_date]
            self.test_Xtime = all_X_pre[mask_at_test_dates]
        
        # Assign target
        tar_all = meta_pl["target_ratio"].to_numpy().flatten()
        
        rat_inbetween = tar_all[mask_inbetween_date]
        rat_at_test_date = tar_all[mask_at_test_dates]
        
        if self.group_type == "Tree":
            self.train_ytree = np.clip(rat_inbetween, 1e-2, 1e2)
            self.test_ytree = np.clip(rat_at_test_date, 1e-2, 1e2)
        
        if self.group_type == "Time":
            inc_factor = self.params["LoadupSamples_time_inc_factor"]
            self.train_ytime = np.tanh((rat_inbetween - 1.0) * inc_factor) / 2.0 + 0.5
            self.test_ytime = np.tanh((rat_at_test_date - 1.0) * inc_factor) / 2.0 + 0.5
        
        # Clean Data
        self.__remove_nan_samples_train()
        
        # Scaling Tree
        if self.group_type == "Tree" and self.params["LoadupSamples_tree_scaling_standard"]:
            self.__scale_tree_standard()
        
        # Scaling Time
        if self.group_type == "Time" and self.params["LoadupSamples_time_scaling_stretch"]:
            self.train_Xtime = self.__scale_time_stretch(self.train_Xtime)
            self.test_Xtime = self.__scale_time_stretch(self.test_Xtime)
            
        # Final checks
        self.dataset_tests()
        
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

        self.test_Xtree = np.concatenate(
            [
                self.train_Xtree[train_split_te_mask], 
                self.test_Xtree[test_split_mask]
            ], 
            axis = 0
        ) if self.test_Xtree is not None and self.train_Xtree is not None else None
        self.test_Xtime = np.concatenate(
            [
                self.train_Xtime[train_split_te_mask], 
                self.test_Xtime[test_split_mask]
            ], 
            axis = 0
        ) if self.test_Xtime is not None and self.train_Xtime is not None else None
        self.test_ytree = np.concatenate(
            [
                self.train_ytree[train_split_te_mask], 
                self.test_ytree[test_split_mask]
            ], 
            axis = 0
        ) if self.test_ytree is not None and self.train_ytree is not None else None
        self.test_ytime = np.concatenate(
            [
                self.train_ytime[train_split_te_mask], 
                self.test_ytime[test_split_mask]
            ], 
            axis = 0
        ) if self.test_ytime is not None and self.train_ytime is not None else None
        
        self.train_Xtree = self.train_Xtree[train_split_tr_mask] if self.train_Xtree is not None else None
        self.train_Xtime = self.train_Xtime[train_split_tr_mask] if self.train_Xtime is not None else None
        self.train_ytree = self.train_ytree[train_split_tr_mask] if self.train_ytree is not None else None
        self.train_ytime = self.train_ytime[train_split_tr_mask] if self.train_ytime is not None else None

        self.meta_pl_test = pl.concat([
            self.meta_pl_train.filter(train_split_te_mask),
            self.meta_pl_test.filter(test_split_mask)
        ])
        self.meta_pl_train = self.meta_pl_train.filter(train_split_tr_mask)

        self.min_test_date = self.meta_pl_test.select(pl.min('date')).item()
        self.test_dates = self.meta_pl_test.select(pl.col('date')).to_series().unique().to_list()
        
        self.dataset_tests()
        
        
    def __add_target(self, meta_pl: pl.DataFrame):
        idx_after = self.__determine_idx_after(meta_pl)
        
        meta_pl = meta_pl.sort(["date", "ticker"])
        meta_pl = meta_pl.with_row_count(name="row_idx")

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
            .shift(-(idx_after-(idx_after//2)))
            .rolling_mean(window_size=idx_after)
            .over("ticker")         #TODO: two expr after over seems to lead to wrong results. Make two separate expressions.
            .alias("target_mean_close")
        )

        # get max over all future close prices after idx_after days
        max_expr = (
            pl.col("AdjClose")
            .shift(-1)
            .rolling_max(window_size=idx_after)
            .over("ticker")         #TODO: two expr after over seems to lead to wrong results. Make two separate expressions.
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
        # Every sample with nan in target are removed
        if self.group_type == "Tree":
            # Remove nan in y
            mask_nan_inbetween_tree = np.isnan(self.train_ytree) 
            if mask_nan_inbetween_tree.sum() > 0:
                logger.warning(f"NaN values found in training tree features. {mask_nan_inbetween_tree.sum()} Samples removed.")
            self.train_Xtree = self.train_Xtree[~mask_nan_inbetween_tree]
            self.train_ytree = self.train_ytree[~mask_nan_inbetween_tree]
            self.meta_pl_train = self.meta_pl_train.filter(~mask_nan_inbetween_tree)
            
            # Remove nan and inf in X
            mask_nan_inbetween_tree = np.isnan(self.train_Xtree).any(axis=1)
            mask_inf_inbetween_tree = np.isinf(self.train_Xtree).any(axis=1)
            mask_isbig_inbetween_tree = (np.abs(self.train_Xtree) > 1e10).any(axis=1)
            mask_combined_inbetween_tree = mask_nan_inbetween_tree | mask_inf_inbetween_tree | mask_isbig_inbetween_tree
            if mask_combined_inbetween_tree.sum() > 0:
                logger.warning(f"Inf values found in training tree features. {mask_combined_inbetween_tree.sum()} Samples removed.")
            self.train_Xtree = self.train_Xtree[~mask_combined_inbetween_tree]
            self.train_ytree = self.train_ytree[~mask_combined_inbetween_tree]
            self.meta_pl_train = self.meta_pl_train.filter(~mask_combined_inbetween_tree)

        if self.group_type == "Time":
            mask_nan_inbetween_time = np.isnan(self.train_ytime)
            if mask_nan_inbetween_time.sum() > 0:
                logger.warning(f"NaN values found in training time features. {mask_nan_inbetween_time.sum()} Samples removed.")
            self.train_Xtime = self.train_Xtime[~mask_nan_inbetween_time]
            self.train_ytime = self.train_ytime[~mask_nan_inbetween_time]
            self.meta_pl_train = self.meta_pl_train.filter(~mask_nan_inbetween_time)
        
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