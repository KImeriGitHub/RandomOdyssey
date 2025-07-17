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
            params: dict = None,
        ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.group = group
        self.train_start_date = train_start_date
        self.test_dates = test_dates
        
        self.min_test_date = min(self.test_dates)
        self.max_test_date = max(self.test_dates)
        
        # Assign parameters to instance variables
        
        self.daysAfter = self.params.get('daysAfterPrediction', None)
        self.idxAfter = self.params.get('idxAfterPrediction', None)
        self.timesteps = self.params['timesteps']
        self.target_option = self.params['target_option']
        
        self.featureTreeNames: list
        self.featureTimeNames: list
        self.meta_pl_train: pl.DataFrame
        self.meta_pl_test: pl.DataFrame
        self.test_Xtree: np.array
        self.test_Xtime: np.array
        self.test_ytree: np.array
        self.test_ytime: np.array
        self.train_Xtree: np.array
        self.train_Xtime: np.array
        self.train_ytree: np.array
        self.train_ytime: np.array
        
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
        namestree: list = []
        namestime: list = []
        all_Xtree_list = []
        all_Xtime_list = []
        
        # check whether files are there otherwise make error
        for year in years:
            label = str(year)
            if not os.path.isfile(main_path + f"TreeFeatures_{label}_{self.group}.npz"):
                raise FileNotFoundError(f"Tree feature sets not found.")
            if not os.path.isfile(main_path + f"TimeFeatures_{label}_{self.group}.npz"):
                raise FileNotFoundError(f"Time feature sets not found.")
            
        # Load data
        testdate_in_db = np.zeros(len(self.test_dates), dtype=bool)
        for year in years:
            label = str(year)
            
            tree_npz = main_path + f"TreeFeatures_{label}_{self.group}.npz"
            time_npz = main_path + f"TimeFeatures_{label}_{self.group}.npz"
            
            data_tree = np.load(tree_npz, allow_pickle=True)
            data_time = np.load(time_npz, allow_pickle=True)
            
            meta_tree  = data_tree['meta_tree']
            
            feats_tree = data_tree['treeFeatures']
            feats_time = data_time['timeFeatures']
            
            namestree = data_tree['treeFeaturesNames']
            namestime = data_time['timeFeaturesNames']
            
            for idx, test_date in enumerate(self.test_dates):
                if test_date in meta_tree["date"]:
                    testdate_in_db[idx] = True
            
            # shorten to timesteps
            if feats_time.shape[1] > self.timesteps:
                feats_time = feats_time[:,-self.timesteps:,:]
            
            # apply masks
            meta_pl_all_loop: pl.DataFrame = pl.DataFrame({
                "date":   meta_tree["date"],
                "ticker": meta_tree["ticker"],
                "Close":  meta_tree["Close"],
            })
            
            meta_all_pl_list.append(meta_pl_all_loop)
            all_Xtree_list.append(feats_tree)
            all_Xtime_list.append(feats_time)
            
        # Post processing
        meta_pl: pl.DataFrame = pl.concat(meta_all_pl_list)
        
        for idx, test_date in enumerate(self.test_dates):
            if not testdate_in_db[idx]:
                logger.warning(f"Test date {test_date} not found in the database. Resetting to last trading day.")
                self.test_dates[idx] = meta_pl.filter(pl.col("date") <= test_date).select("date").max()["date"].item()
        
        self.featureTreeNames = namestree
        self.featureTimeNames = namestime
        all_Xtree_pre = np.concatenate(all_Xtree_list, axis=0)
        all_Xtime_pre = np.concatenate(all_Xtime_list, axis=0)  
        
        # Add target
        meta_pl = meta_pl.sort(["date", "ticker"])
        meta_pl = self.__add_target(meta_pl)
        meta_pl = meta_pl.sort(["date", "ticker"])
        meta_pl = meta_pl.with_columns((pl.col("target_close")  / pl.col("Close")).alias("target_ratio"))
        
        # Assign Main Variables
        mask_at_test_dates = (meta_pl["date"].is_in(self.test_dates)).fill_null(False).to_numpy()
        mask_inbetween_date = (
            (meta_pl["date"] >= self.train_start_date)
            & (meta_pl["target_date"] < self.min_test_date)
        ).fill_null(False).to_numpy()
        
        self.meta_pl_train = meta_pl.filter(mask_inbetween_date)
        self.meta_pl_test  = meta_pl.filter(mask_at_test_dates)
        
        self.train_Xtree = all_Xtree_pre[mask_inbetween_date]
        self.train_Xtime = all_Xtime_pre[mask_inbetween_date]
        
        self.test_Xtree = all_Xtree_pre[mask_at_test_dates]
        self.test_Xtime = all_Xtime_pre[mask_at_test_dates]
        
        # Assign target
        tar_all = meta_pl["target_close"].to_numpy().flatten()
        cur_all = meta_pl["Close"].to_numpy().flatten()
        
        rat_at_test_date = tar_all[mask_at_test_dates] / cur_all[mask_at_test_dates]
        rat_inbetween = tar_all[mask_inbetween_date] / cur_all[mask_inbetween_date]
        
        inc_factor = self.params["LoadupSamples_time_inc_factor"]
        self.train_ytree = np.clip(rat_inbetween, 1e-1, 1e1)
        self.train_ytime = np.tanh((rat_inbetween - 1.0)*inc_factor)/2.0 + 0.5
        
        self.test_ytree = np.clip(rat_at_test_date, 1e-1, 1e1)
        self.test_ytime = np.tanh((rat_at_test_date - 1.0)*inc_factor)/2.0 + 0.5
        
        # Clean Data
        self.__remove_nan_samples_train()
        
        # Scaling Tree
        if self.params["LoadupSamples_tree_scaling_standard"]:
            self.__scale_tree_standard()
        
        # Scaling Time
        if self.params["LoadupSamples_time_scaling_stretch"]:
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
        
        self.test_Xtree = np.concatenate(
            [
                self.train_Xtree[train_split_te_mask], 
                self.test_Xtree
            ], 
            axis = 0
        )
        self.test_Xtime = np.concatenate(
            [
                self.train_Xtime[train_split_te_mask], 
                self.test_Xtime
            ], 
            axis = 0
        )
        self.test_ytree = np.concatenate(
            [
                self.train_ytree[train_split_te_mask], 
                self.test_ytree
            ], 
            axis = 0
        )
        self.test_ytime = np.concatenate(
            [
                self.train_ytime[train_split_te_mask], 
                self.test_ytime
            ], 
            axis = 0
        )
        
        self.train_Xtree = self.train_Xtree[train_split_tr_mask]
        self.train_Xtime = self.train_Xtime[train_split_tr_mask]
        self.train_ytree = self.train_ytree[train_split_tr_mask]
        self.train_ytime = self.train_ytime[train_split_tr_mask]
        
        self.meta_pl_test = pl.concat([
            self.meta_pl_train.filter(train_split_te_mask),
            self.meta_pl_test
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

        allclose_exprs = ([
            pl.col("Close")
            .shift(-i)
            .over("ticker")
            .alias(f"target_close_at{i}")
            for i in range(1, idx_after + 1)
        ])
        
        # get corresponding row indices for all future close prices after idx_after days
        allrowidx_exprs = ([
            pl.col("row_idx")
            .shift(-i)
            .over("ticker")
            .alias(f"target_rowidx_at{i}")
            for i in range(1, idx_after + 1)
        ])

        # get target close price after idx_after days
        last_expr = (
            pl.col("Close")
            .shift(-idx_after)
            .over("ticker")
            .alias("target_last_close")
        )

        # get mean over all future close prices after idx_after days
        mean_expr = (
            pl.col("Close")
            .shift(-1)
            .over("ticker")
            .rolling_mean(window_size=idx_after)
            .alias("target_mean_close")
        )

        # get max over all future close prices after idx_after days
        max_expr = (
            pl.col("Close")
            .shift(-1)
            .over("ticker")
            .rolling_max(window_size=idx_after)
            .alias("target_max_close")
        )
        
        if self.target_option == 'last':
            option_expr = last_expr.alias("target_close")
        elif self.target_option == 'mean':
            option_expr = mean_expr.alias("target_close")
        elif self.target_option == 'max':
            option_expr = max_expr.alias("target_close")
        else:
            raise ValueError(f"Unknown target option: {self.target_option}. Choose from 'last', 'mean', or 'max'.")

        return meta_pl.with_columns(
            [date_expr] + 
            allclose_exprs + allrowidx_exprs + 
            [last_expr] + [mean_expr] + [max_expr] + 
            [option_expr]
        )
    
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
        mask_nan_inbetween_tree = np.isnan(self.train_ytree) 
        mask_nan_inbetween_time = np.isnan(self.train_ytime)
        
        if mask_nan_inbetween_tree.sum() > 0 or mask_nan_inbetween_time.sum() > 0:
            logger.warning(f"NaN values found in training targets. {mask_nan_inbetween_tree.sum()} Samples removed.")
        else:
            return

        self.train_Xtree = self.train_Xtree[~mask_nan_inbetween_tree]
        self.train_Xtime = self.train_Xtime[~mask_nan_inbetween_time]
        
        self.train_ytree = self.train_ytree[~mask_nan_inbetween_tree]
        self.train_ytime = self.train_ytime[~mask_nan_inbetween_time]
        
        self.meta_pl_train = self.meta_pl_train.filter(~mask_nan_inbetween_tree)
        
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
    
    
# Code to use to implement filterting during loadup
"""
    testdate_in_db = np.zeros(len(self.test_dates), dtype=bool)
    for year in years:
        label = str(year)

        tree_npz = main_path + f"TreeFeatures_{label}_{self.group}.npz"
        time_npz = main_path + f"TimeFeatures_{label}_{self.group}.npz"
        data_tree = np.load(tree_npz, allow_pickle=True)
        data_time = np.load(time_npz, allow_pickle=True)
        meta_tree  = data_tree['meta_tree']
        feats_tree = data_tree['treeFeatures']
        names_tree = data_tree['treeFeaturesNames']
        #meta_time  = data_time['meta_time']
        feats_time = data_time['timeFeatures']
        names_time = data_time['timeFeaturesNames']

        for idx, test_date in enumerate(self.test_dates):
            if test_date in meta_tree["date"]:
                testdate_in_db[idx] = True

        # shorten to timesteps
        if feats_time.shape[1] > self.timesteps:
            feats_time = feats_time[:,-self.timesteps:,:]

        mask = self.__forge_filtermask(feats_tree, names_tree)

        # apply masks
        meta_pl_all_loop: pl.DataFrame = pl.DataFrame({
            "date":   meta_tree["date"],
            "ticker": meta_tree["ticker"],
            "Close":  meta_tree["Close"],
        })

        namestree = names_tree
        namestime = names_time
        mask_list.append(mask)
        meta_all_pl_list.append(meta_pl_all_loop)
        all_Xtree_list.append(feats_tree[mask])
        all_Xtime_list.append(feats_time[mask])
    mask_all = np.concatenate(mask_list)
    meta_pl_all: pl.DataFrame = pl.concat(meta_all_pl_list)      
    self.featureTreeNames = namestree
    self.featureTimeNames = namestime
    all_Xtree_pre = np.concatenate(all_Xtree_list, axis=0)
    all_Xtime_pre = np.concatenate(all_Xtime_list, axis=0)  

    for idx, test_date in enumerate(self.test_dates):
        if not testdate_in_db[idx]:
            logger.warning(f"Test date {test_date} not found in the database. Resetting to last trading day.")
            self.test_dates[idx] = meta_pl_all.filter(pl.col("date") <= test_date).select("date").max()["date"].item()

# get target value
meta_pl_all = self.__add_target(meta_pl_all, self.daysAfter)
meta_pl_all = meta_pl_all.with_columns((pl.col("target_close")  / pl.col("Close")).alias("target_ratio"))
tar_all = meta_pl_all["target_close"].to_numpy().flatten()
cur_all = meta_pl_all["Close"].to_numpy().flatten()
tar_masked = tar_all[mask_all]
cur_masked = cur_all[mask_all]
meta_pl = meta_pl_all.filter(mask_all)

def __forge_filtermask(self, feats_tree: np.array, names_tree: np.array) -> np.array:
    mask = np.zeros(feats_tree.shape[0], dtype=bool)
    
    if not self.params["TreeTime_isFiltered"]:
        return np.ones(feats_tree.shape[0], dtype=bool)
    
    if self.params.get("TreeTime_FourierRSME_q", None) is not None:
        idx = np.where(names_tree == 'Fourier_Price_RSMERatioCoeff_1_MH_2')[0]
        arr = feats_tree[:, idx].flatten()
        quant = np.quantile(arr, self.params["TreeTime_FourierRSME_q"])
        mask = mask | (arr <= quant)
        
    if self.params.get("TreeTime_trend_stc_q", None) is not None:
        idx = np.where(names_tree == 'FeatureTA_trend_stc')[0]
        arr = feats_tree[:, idx].flatten()
        quant_lower = np.quantile(arr, self.params["TreeTime_trend_stc_q"])
        mask = mask | (arr <= quant_lower)
        
    if self.params.get("TreeTime_trend_mass_index_q", None) is not None:
        idx = np.where(names_tree == 'FeatureTA_trend_mass_index')[0]
        arr = feats_tree[:, idx].flatten()
        quant_lower = np.quantile(arr, self.params["TreeTime_trend_mass_index_q"])
        mask = mask | (arr <= quant_lower)
        
    if self.params.get("TreeTime_AvgReturnPct_qup", None) is not None:
        idx = np.where(names_tree == 'FeatureGroup_AvgReturnPct')[0]
        arr = feats_tree[:, idx].flatten()
        quant = np.quantile(arr, self.params["TreeTime_AvgReturnPct_qup"])
        mask = mask | (arr >= quant)
        
    if self.params.get("TreeTime_volatility_atr_qup", None) is not None:
        idx = np.where(names_tree == 'FeatureTA_volatility_atr')[0]
        arr = feats_tree[:, idx].flatten()
        quant = np.quantile(arr, self.params["TreeTime_volatility_atr_qup"])
        mask = mask | (arr >= quant)
        
    if self.params.get("TreeTime_ReturnLog_RSMECoeff_2_qup", None) is not None:
        lag = int(self.daysAfter * (5/7))
        candidates = [
                f'Fourier_ReturnLog_RSMECoeff_2_MH_{m}_lag_m{lag}' for m in [4,5,6,7,8]
            ] + [
                f'Fourier_ReturnLog_RSMECoeff_2_MH_{m}_lag_m{lag+1}' for m in [4,5,6,7,8]
            ]
        name = next((c for c in candidates if c in names_tree), None)
        if name is not None:
            idx = np.where(names_tree == name)[0]
            arr = feats_tree[:, idx].flatten()
            quant = np.quantile(arr, self.params["TreeTime_ReturnLog_RSMECoeff_2_qup"])
            mask = mask | (arr >= quant)
        else:
            logger.warning("No Fourier ReturnLog RSME Coeff feature found in the dataset. Skipping filtering by Fourier ReturnLog RSME Coeff.")
        
    if self.params.get("TreeTime_Drawdown_q", None) is not None:
        candidates = [
            'MathFeature_Drawdown_MH4',
            'MathFeature_Drawdown_MH5',
            'MathFeature_Drawdown_MH6',
        ]
        name = next((c for c in candidates if c in names_tree), None)
        if name is not None:
            idx = np.where(names_tree == name)[0]
            arr = feats_tree[:, idx].flatten()
            quant_lower = np.quantile(arr, self.params["TreeTime_Drawdown_q"])
            mask = mask | (arr <= quant_lower)
        else:
            logger.warning("No Drawdown feature found in the dataset. Skipping filtering by Drawdown.")
        
    if mask.sum() == 0:
        raise ValueError("No features were selected by filtering.")
    
    return mask

"""