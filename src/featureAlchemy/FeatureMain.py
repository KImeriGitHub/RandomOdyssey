import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List
import itertools

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.mathTools.TAIndicators import TAIndicators
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

from src.featureAlchemy.FeatureFourierCoeff import FeatureFourierCoeff
from src.featureAlchemy.FeatureCategory import FeatureCategory
from src.featureAlchemy.FeatureFinancialData import FeatureFinancialData
from src.featureAlchemy.FeatureMathematical import FeatureMathematical
from src.featureAlchemy.FeatureSeasonal import FeatureSeasonal
from src.featureAlchemy.FeatureTA import FeatureTA
from src.featureAlchemy.FeatureGroupDynamic import FeatureGroupDynamic

class FeatureMain():
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 15,
        'multFactor': 8,
        'monthsHorizon': 13,
        'timesteps': 15,
    }

    def __init__(self, 
                 assets: Dict[str, AssetDataPolars],
                 startDate: pd.Timestamp, 
                 endDate:pd.Timestamp, 
                 lagList: List[int],
                 monthHorizonList: List[int],
                 params: dict = None):
        
        self.assets = assets
        self.startDate = startDate
        self.endDate = endDate
        self.lagList = lagList
        self.monthHorizonList = monthHorizonList
        
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.monthsHorizon = self.params['monthsHorizon']
        self.timesteps = self.params['timesteps']
        
        # get business days
        self.tickers = list(self.assets.keys())
        self.nAssets = len(self.tickers)
        exampleAsset = self.assets[self.tickers[0]]
        
        #preprocess
        exampleAsset_start_idx = DPl(exampleAsset.shareprice).getNextLowerOrEqualIndex(self.startDate)
        exampleAsset_start_idx = exampleAsset_start_idx if exampleAsset.shareprice["Date"].item(exampleAsset_start_idx) == self.startDate else exampleAsset_start_idx + 1
        exampleAsset_end_idx = DPl(exampleAsset.shareprice).getNextLowerOrEqualIndex(self.endDate)
        self.business_days= exampleAsset.shareprice["Date"].slice(exampleAsset_start_idx, exampleAsset_end_idx - exampleAsset_start_idx + 1).to_numpy()  
        self.business_days = np.array([pd.Timestamp(x, tz ="UTC") for x in self.business_days])      
        self.startBDate = self.business_days[0]
        self.endBDate = self.business_days[-1]
        self.nDates = len(self.business_days)
        
        self.nAssets = len(self.assets)
        
        self.idxAssets_start = {}
        self.idxAssets_end = {}
        for ticker, _ in self.assets.items():
            self.idxAssets_start[ticker] = DPl(self.assets[ticker].shareprice).getNextLowerOrEqualIndex(self.startBDate)
            self.idxAssets_end[ticker] = DPl(self.assets[ticker].shareprice).getNextLowerOrEqualIndex(self.endBDate)
            
        assert np.all([(self.idxAssets_end[ticker]-self.idxAssets_start[ticker]+1) == self.nDates for ticker in self.idxAssets_start.keys()]), "All assets must have the same number of dates."
        
        self.FGD = FeatureGroupDynamic(self.assets, self.startBDate, self.endBDate, self.lagList, self.monthHorizonList, self.params)
        print(f"FeatureMain initialized with {self.nAssets} assets and {self.nDates} dates.")
    
    def getTreeFeatures(self) -> pl.DataFrame:
        # 1) gather feature‑name lists
        exampleAsset = self.assets[self.tickers[0]]
        featureNames = [
            FeatureCategory(exampleAsset, self.params).getFeatureNames(),
            FeatureMathematical(exampleAsset, self.lagList, self.monthHorizonList, self.params).getFeatureNames(),
            FeatureFourierCoeff(exampleAsset, self.startBDate, self.endBDate, self.lagList, self.monthHorizonList, self.params).getFeatureNames(),
            FeatureFinancialData(exampleAsset, self.lagList, self.params).getFeatureNames(),
            FeatureSeasonal(exampleAsset, self.startBDate, self.endBDate, self.lagList, self.params).getFeatureNames(),
            FeatureTA(exampleAsset, self.startBDate, self.endBDate, self.lagList, self.params).getFeatureNames(),
            self.FGD.getFeatureNames()
        ]
        # flatten into one long list
        flat_features = list(itertools.chain.from_iterable(featureNames))
        
        # 2) preallocate arrays
        nD, nA, nF = self.nDates, self.nAssets, len(flat_features)
        arr     = np.empty((nD, nA, nF), dtype=np.float32)
        
        # 3) fill arrays
        niveau = 1.0
        idx_dict = self.idxAssets_start.copy()
        FGD_tick_date = {date: self.FGD.apply(date, {k: v + i for k, v in idx_dict.items()}) for i, date in enumerate(self.business_days)}
        for tidx, ticker in enumerate(self.assets.keys()):
            print(f"  Processing ticker {ticker} ({tidx+1}/{self.nAssets})")
            FC = FeatureCategory(self.assets[ticker], self.params)
            FM = FeatureMathematical(self.assets[ticker], self.lagList, self.monthHorizonList, self.params)
            FFC = FeatureFourierCoeff(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.monthHorizonList, self.params)
            FFD = FeatureFinancialData(self.assets[ticker], self.lagList, self.params)
            FS = FeatureSeasonal(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.params)
            FTA = FeatureTA(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.params)
            idx_dict = {k: v + 1 for k, v in idx_dict.items()}
            for date_idx, date in enumerate(self.business_days):
                print(f"Processing date {date} ({date_idx+1}/{self.nDates})")
                arr_loop = [
                    FC.apply(niveau),
                    FM.apply(date, niveau, self.idxAssets_start[ticker] + date_idx),
                    FFC.apply(date, niveau, self.idxAssets_start[ticker] + date_idx),
                    FFD.apply(date, niveau, self.idxAssets_start[ticker] + date_idx),
                    FS.apply(date, niveau),
                    FTA.apply(date, niveau, self.idxAssets_start[ticker] + date_idx),
                    FGD_tick_date[date][ticker]
                ]
                arr[date_idx, tidx, :] = np.concatenate(arr_loop)
                
        # 4) reshape to (nD*nA, nF) and build Polars DataFrame
        arr_flat = arr.reshape(nD * nA, nF)
        dates    = np.repeat(self.business_days, self.nAssets)
        tickers  = np.tile(list(self.assets.keys()), self.nDates)
        close    = np.array([
            self.assets[ticker].shareprice["Close"].item(self.idxAssets_start[ticker] + i)
            for ticker in self.assets.keys() for i in range(self.nDates)
        ])

        df = pl.DataFrame({
            "date":   dates,
            "ticker": tickers,
            "close": close,
            **{feat: arr_flat[:, i] for i, feat in enumerate(flat_features)}
        })
        
        return df
        
    def getTimeFeatures(self) -> tuple[np.array, np.array]:
        # 1) gather feature‑name lists
        exampleAsset = next(iter(self.assets.values()))
        featureNames = [
            FeatureCategory(exampleAsset, self.params).getTimeFeatureNames(),
            FeatureMathematical(exampleAsset, self.lagList, self.monthHorizonList, self.params).getTimeFeatureNames(),
            FeatureFourierCoeff(exampleAsset, self.startBDate, self.endBDate, self.lagList, self.monthHorizonList, self.params).getTimeFeatureNames(),
            FeatureFinancialData(exampleAsset, self.lagList, self.params).getTimeFeatureNames(),
            FeatureSeasonal(exampleAsset, self.startBDate, self.endBDate, self.lagList, self.params).getTimeFeatureNames(),
            FeatureTA(exampleAsset, self.startBDate, self.endBDate, self.lagList, self.params).getTimeFeatureNames(),
            self.FGD.getTimeFeatureNames()
        ]
        # flatten into one long list
        flat_features = list(itertools.chain.from_iterable(featureNames))
        
        # 2) preallocate arrays
        nD, nA, nT, nF = self.nDates, self.nAssets, self.timesteps, len(flat_features)
        arr     = np.empty((nD, nA, nT, nF), dtype=np.float32)
        metaarr = np.empty((nD, nA, 3), dtype=object)  # [date,ticker,close]
        
        # 3) fill arrays
        idx_dict = self.idxAssets_start.copy()
        FGD_tick_date = {date: self.FGD.apply_timeseries(date, {k: v + i for k, v in idx_dict.items()}) for i, date in enumerate(self.business_days)}
        for tidx, ticker in enumerate(self.assets.keys()):
            print(f"  Processing ticker {ticker} ({tidx+1}/{self.nAssets})")
            FC = FeatureCategory(self.assets[ticker], self.params)
            FM = FeatureMathematical(self.assets[ticker], self.lagList, self.monthHorizonList, self.params)
            FFC = FeatureFourierCoeff(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.monthHorizonList, self.params)
            FFD = FeatureFinancialData(self.assets[ticker], self.lagList, self.params)
            FS = FeatureSeasonal(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.params)
            FTA = FeatureTA(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.params)
            idx_dict = {k: v + 1 for k, v in idx_dict.items()}
            for date_idx, date in enumerate(self.business_days):
                print(f"Processing date {date} ({date_idx+1}/{self.nDates})")
                metaarr[date_idx, tidx, 0] = date
                metaarr[date_idx, tidx, 1] = ticker
                metaarr[date_idx, tidx, 2] = self.assets[ticker].shareprice["Close"].item(self.idxAssets_start[ticker] + date_idx)

                arr_loop = [
                    FC.apply_timeseries(date),
                    FM.apply_timeseries(date, self.idxAssets_start[ticker] + date_idx),
                    FFC.apply_timeseries(date, self.idxAssets_start[ticker] + date_idx),
                    FFD.apply_timeseries(date, self.idxAssets_start[ticker] + date_idx),
                    FS.apply_timeseries(date),
                    FTA.apply_timeseries(date, self.idxAssets_start[ticker] + date_idx),
                    FGD_tick_date[date][ticker]
                ]

                arr[date_idx, tidx, :, :] = np.hstack(arr_loop)  # (nT, nF)
                
        metaarr = metaarr.reshape(nD * nA, 3)
        arr = arr.reshape(nD * nA, nT, nF)
        
        return metaarr, arr, flat_features