import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Type
import itertools
import datetime
from itertools import chain

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.mathTools.TAIndicators import TAIndicators
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

from src.featureAlchemy.IFeature import IFeature
from src.featureAlchemy.FeatureFourierCoeff import FeatureFourierCoeff
from src.featureAlchemy.FeatureCategory import FeatureCategory
from src.featureAlchemy.FeatureFinancialData import FeatureFinancialData
from src.featureAlchemy.FeatureMathematical import FeatureMathematical
from src.featureAlchemy.FeatureSeasonal import FeatureSeasonal
from src.featureAlchemy.FeatureTA import FeatureTA
from src.featureAlchemy.FeatureGroupDynamic import FeatureGroupDynamic

import logging
logger = logging.getLogger(__name__)

class MainFeature():
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 15,
        'multFactor': 8,
        'timesteps': 30,
        'lagList': [1, 2, 5, 10, 20, 50, 100, 200, 300, 500],
        'monthHorizonList': [1, 2, 4, 6, 8, 12],
    }

    def __init__(self, 
            assets: Dict[str, AssetDataPolars],
            feature_classes: List[Type[IFeature]],
            startDate: datetime.date, 
            endDate: datetime.date,
            params: dict = None
        ):
        
        self.assets = assets
        self.feature_classes = feature_classes
        self.startDate = startDate
        self.endDate = endDate
        
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.timesteps = self.params['timesteps']
        self.lagList = self.params['lagList']
        self.monthHorizonList = self.params['monthHorizonList']
        
        # get business days
        self.tickers = list(self.assets.keys())
        self.nAssets = len(self.tickers)
        
        #preprocess
        self.business_days= pd.bdate_range(start=self.startDate, end=self.endDate).date.tolist()
        self.trading_days = self._calculate_trading_days(self.business_days)
        self.nDates = len(self.trading_days)
        self.startTDate = self.trading_days[0]
        self.endTDate = self.trading_days[-1]
        
        self.idxAssets = {ticker: DOps(self.assets[ticker].shareprice).getNextLowerOrEqualIndices(self.trading_days) for ticker in self.assets.keys()}
        self.idxAssets_at = {ticker: DOps(self.assets[ticker].shareprice).getIndices(self.trading_days) for ticker in self.assets.keys()}
        
        self.FGD = FeatureGroupDynamic(self.assets, self.startTDate, self.endTDate, self.params)
        logger.info(f"  FeatureMain initialized with {self.nAssets} assets and {self.nDates} dates.")
    
    def _calculate_trading_days(self, business_days: list[datetime.date]) -> list[datetime.date]:
        """
        Returns the actual trading days from asset data that correspond to a 
        given list of business days.
        """
        # For each ticker, find the indices of the last available share price date 
        # for each requested business day.
        indices_by_ticker: Dict[str, List[int]] = {
            ticker: DOps(self.assets[ticker].shareprice).getNextLowerOrEqualIndices(business_days) 
            for ticker in self.tickers
        }

        # Create a generator of all trading date lists from all assets.
        all_trading_dates_nested = (
            self.assets[ticker].shareprice['Date'].gather(indices_by_ticker[ticker]).to_list()
            for ticker in self.tickers
        )
        
        # Flatten the nested lists, find the unique dates, and return them sorted.
        unique_dates = {d for d in chain.from_iterable(all_trading_dates_nested)}
        
        return sorted(unique_dates)
    
    def get_features(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        nD, nA = self.nDates, self.nAssets
        tickers = list(self.assets.keys())

        # --- Meta Information Array ---

        # 1) Define the dtype for the structured array to hold meta information
        dtype = [
            ("date",     "datetime64[D]"),
            ("ticker",   "U10"),
            ("Close",    "f8"),
            ("AdjClose", "f8"),
            ("Open",     "f8"),
        ]  # Fields: date, ticker, Close, AdjClose, Open

        # 2) Allocate the structured array for meta information
        metaarr = np.zeros((nD, nA), dtype=dtype)

        # 3) Assign dates and tickers (vectorized)
        metaarr["date"]   = np.array(self.trading_days)[:, None]
        metaarr["ticker"] = np.array(tickers)[None, :]

        # 4) Assign price columns by gathering from each asset's shareprice DataFrame
        metaarr["Close"] = np.array([
            a.shareprice["Close"].gather(self.idxAssets_at[t]).to_numpy()
            for t, a in (self.assets.items())
        ]).transpose()
        metaarr["AdjClose"] = np.array([
            a.shareprice["AdjClose"].gather(self.idxAssets_at[t]).to_numpy()
            for t, a in (self.assets.items())
        ]).transpose()
        metaarr["Open"] = np.array([
            a.shareprice["Open"].gather(self.idxAssets_at[t]).to_numpy()
            for t, a in (self.assets.items())
        ]).transpose()

        # 5) Create a mask: True where all price columns are not NaN, False otherwise
        mask = np.ones((nD, nA), dtype=bool)
        fields_list_containing_nan = ["Close", "AdjClose", "Open"]
        for field in fields_list_containing_nan:
            mask &= ~np.isnan(metaarr[field])

        # --- Feature Array ---
        arr_list = []
        featureNames: list[str] = []

        for Cls in self.feature_classes:
            logger.info(f"  Processing {Cls.__name__}")
            feat_name = Cls.__name__

            # For GroupDynamic, Cls(...) already takes all assets; others take one asset at a time
            if feat_name in ['FeatureGroupDynamic', 'FeatureGroupDynamicTS']:
                inst: IFeature = Cls(self.assets, startDate = self.startTDate, endDate = self.endTDate, params = self.params)
                names = inst.getFeatureNames()
                mat_dict = inst.apply(self.trading_days)                      # for each ticker (nD, ..., nF)
                stack = np.stack([mat_dict[t] for t in tickers], axis=1)      # (nD, nA, ..., nF)
            
            else:
                loop_mats = []
                for t in tickers:
                    inst = Cls(self.assets[t], startDate = self.startTDate, endDate = self.endTDate, params = self.params)
                    names = inst.getFeatureNames()
                    loop_mats.append(inst.apply(self.trading_days))  # each is (nD, ..., nF)
                stack = np.stack(loop_mats, axis=1)                  # (nD, nA, ..., nF)

            featureNames.extend(names)
            arr_list.append(stack)

        arr = np.concatenate(arr_list, axis=-1)  # (nD, nA, ..., nF)
        metaarr = metaarr[mask]   # this reshapes to (nD*nA) but with valid inputs
        arr = arr[mask]     # this reshapes to (nD*nA, ..., nF) where nD*nA is not correct its with valid inputs
        
        return metaarr, arr, featureNames
        
    def getTreeFeatures(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        # 1) gather feature‑name lists
        exampleAsset = self.assets[self.tickers[0]]
        featureNames = [
            FeatureCategory(exampleAsset, self.params).getFeatureNames(),
            FeatureMathematical(exampleAsset, self.params).getFeatureNames(),
            FeatureFourierCoeff(exampleAsset, self.startTDate, self.endTDate, self.params).getFeatureNames(),
            FeatureFinancialData(exampleAsset, self.params).getFeatureNames(),
            FeatureSeasonal(exampleAsset, self.startTDate, self.endTDate, self.params).getFeatureNames(),
            FeatureTA(exampleAsset, self.startTDate, self.endTDate, self.params).getFeatureNames(),
            self.FGD.getFeatureNames()
        ]
        # flatten into one long list
        flat_features = list(itertools.chain.from_iterable(featureNames))
        
        # 2) preallocate arrays
        nD, nA, nF = self.nDates, self.nAssets, len(flat_features)
        arr     = np.empty((nD, nA, nF), dtype=np.float32)
        metaarr = np.empty((nD, nA), dtype=[('date', 'datetime64[D]'), ('ticker', 'U10'), ('Close', 'f4')])  # [date,ticker,close]
        mask = np.ones((nD, nA), dtype=bool)  # mask for missing values
        
        # 3) fill arrays
        niveau = 1.0
        FGD_tick_date = {date: self.FGD.apply(date, {tic: idxlist[i] for tic, idxlist in self.idxAssets.items()}) for i, date in enumerate(self.business_days)}
        for tidx, ticker in enumerate(self.assets.keys()):
            logger.info(f"  Processing ticker {ticker} ({tidx+1}/{self.nAssets})")
            FC = FeatureCategory(self.assets[ticker], self.params)
            FM = FeatureMathematical(self.assets[ticker], self.params)
            FFC = FeatureFourierCoeff(self.assets[ticker], self.startTDate, self.endTDate, self.params)
            FFD = FeatureFinancialData(self.assets[ticker], self.params)
            FS = FeatureSeasonal(self.assets[ticker], self.startTDate, self.endTDate, self.params)
            FTA = FeatureTA(self.assets[ticker], self.startTDate, self.endTDate, self.params)
            idcs = self.idxAssets_at[ticker]
            for date_idx, date in enumerate(self.business_days):
                if idcs[date_idx] is None:
                    mask[date_idx, tidx] = False
                    continue
                
                metaarr[date_idx, tidx]['date'] = date
                metaarr[date_idx, tidx]['ticker'] = ticker
                metaarr[date_idx, tidx]["Close"] = self.assets[ticker].shareprice["Close"].item(idcs[date_idx])

                arr[date_idx, tidx, :] = np.concatenate([
                    FC.apply(niveau),
                    FM.apply(date, niveau, idcs[date_idx]),
                    FFC.apply(date, niveau, idcs[date_idx]),
                    FFD.apply(date, niveau, idcs[date_idx]),
                    FS.apply(date, niveau),
                    FTA.apply(date, niveau, idcs[date_idx]),
                    FGD_tick_date[date][ticker]
                ])
         
        # 4) remove missing values
        metaarr = metaarr[mask]   # this reshapes to (nD*nA) but with valid inputs
        arr = arr[mask]     # this reshapes to (nD*nA, nF) where nD*nA is not correct its with valid inputs
        
        return metaarr, arr, flat_features
        
    def getTimeFeatures(self) -> tuple[np.array, np.array, list[str]]:
        # 1) gather feature‑name lists
        exampleAsset = next(iter(self.assets.values()))
        featureNames = [
            #FeatureCategory(exampleAsset, self.params).getTimeFeatureNames(),
            FeatureMathematical(exampleAsset, self.params).getTimeFeatureNames(),
            FeatureFourierCoeff(exampleAsset, self.startTDate, self.endTDate, self.params).getTimeFeatureNames(),
            FeatureFinancialData(exampleAsset, self.params).getTimeFeatureNames(),
            FeatureSeasonal(exampleAsset, self.startTDate, self.endTDate, self.params).getTimeFeatureNames(),
            FeatureTA(exampleAsset, self.startTDate, self.endTDate, self.params).getTimeFeatureNames(),
            self.FGD.getTimeFeatureNames()
        ]
        # flatten into one long list
        flat_features = list(itertools.chain.from_iterable(featureNames))
        
        # 2) preallocate arrays
        nD, nA, nT, nF = self.nDates, self.nAssets, self.timesteps, len(flat_features)
        arr     = np.empty((nD, nA, nT, nF), dtype=np.float32)
        metaarr = np.empty((nD, nA), dtype=[('date', 'datetime64[D]'), ('ticker', 'U10'), ('Close', 'f4')])  # [date,ticker,close]
        mask = np.ones((nD, nA), dtype=bool)  # mask for missing values
        
        # 3) fill arrays
        FGD_tick_date = {date: self.FGD.apply_timeseries(date, {tic: idxlist[i] for tic, idxlist in self.idxAssets.items()}) for i, date in enumerate(self.business_days)}
        for tidx, ticker in enumerate(self.assets.keys()):
            logger.info(f"  Processing ticker {ticker} ({tidx+1}/{self.nAssets})")
            #FC = FeatureCategory(self.assets[ticker], self.params)
            FM = FeatureMathematical(self.assets[ticker], self.params)
            FFC = FeatureFourierCoeff(self.assets[ticker], self.startTDate, self.endTDate, self.params)
            FFD = FeatureFinancialData(self.assets[ticker], self.params)
            FS = FeatureSeasonal(self.assets[ticker], self.startTDate, self.endTDate, self.params)
            FTA = FeatureTA(self.assets[ticker], self.startTDate, self.endTDate, self.params)
            idcs = self.idxAssets_at[ticker]
            for date_idx, date in enumerate(self.business_days):
                if idcs[date_idx] is None:
                    mask[date_idx, tidx] = False
                    continue
                
                metaarr[date_idx, tidx]['date'] = date
                metaarr[date_idx, tidx]['ticker'] = ticker
                metaarr[date_idx, tidx]["Close"] = self.assets[ticker].shareprice["Close"].item(idcs[date_idx])

                arr[date_idx, tidx, :, :] = np.hstack([ 
                    #FC.apply_timeseries(date),
                    FM.apply_timeseries(date, idcs[date_idx]),
                    FFC.apply_timeseries(date, idcs[date_idx]),
                    FFD.apply_timeseries(date, idcs[date_idx]),
                    FS.apply_timeseries(date),
                    FTA.apply_timeseries(date, idcs[date_idx]),
                    FGD_tick_date[date][ticker]
                ])  # (nT, nF)
                
        # 4) remove missing values
        metaarr = metaarr[mask]   # this reshapes to (nD*nA) but with valid inputs
        arr = arr[mask]     # this reshapes to (nD*nA, nT, nF) where nD*nA is not correct its with valid inputs
        
        return metaarr, arr, flat_features