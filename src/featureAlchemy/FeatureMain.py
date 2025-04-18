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
        self.business_days = pd.date_range(start=self.startDate, end=endDate, freq="B")
        first_bd = self.business_days[0]
        last_bd = self.business_days[-1]
        self.nDates = len(self.business_days)
        
        self.nAssets = len(self.assets)
        
        self.idxAssets_start = {}
        self.idxAssets_end = {}
        for ticker, _ in self.assets.items():
            self.idxAssets_start[ticker] = DPl(self.assets[ticker].shareprice).getNextLowerOrEqualIndex(first_bd)
            self.idxAssets_end[ticker] = DPl(self.assets[ticker].shareprice).getNextLowerOrEqualIndex(last_bd)
            
        assert (
            np.all([(self.idxAssets_end[ticker]-self.idxAssets_start[ticker]+1) == self.nDates for ticker in self.idxAssets_start.keys()]), 
            "All assets must have the same number of dates."
        )
        
        self.FGD = FeatureGroupDynamic(self.assets, self.startDate, self.endDate, self.lagList, self.monthHorizonList, self.params)
    
    def getTreeFeatures(self) -> pl.DataFrame:
        # 1) gather feature‑name lists
        exampleAsset = next(iter(self.assets.values()))
        featureNames = [
            FeatureCategory(exampleAsset).getFeatureNames(),
            FeatureMathematical(exampleAsset, self.lagList, self.monthHorizonList, self.params).getFeatureNames(),
            FeatureFourierCoeff(exampleAsset, self.startDate, self.endDate, self.lagList, self.params).getFeatureNames(),
            FeatureFinancialData(exampleAsset, self.lagList, self.params).getFeatureNames(),
            FeatureSeasonal(exampleAsset, self.startDate, self.endDate, self.lagList, self.params).getFeatureNames(),
            FeatureTA(exampleAsset, self.startDate, self.endDate, self.lagList, self.params).getFeatureNames(),
            self.FGD.getFeatureNames()
        ]
        # flatten into one long list
        flat_features = list(itertools.chain.from_iterable(featureNames))
        
        # 2) preallocate arrays
        nD, nA, nF = self.nDates, self.nAssets, len(flat_features)
        arr     = np.empty((nD, nA, nF), dtype=np.float32)
        
        # 3) fill arrays
        niveau = 1.0
        for date_idx, date in enumerate(self.business_days):
            groupDynDict = self.FGD.apply(date, self.idxAssets_start[ticker]+date_idx)
            for tidx, ticker in enumerate(self.assets.keys()):
                asset = self.assets[ticker]

                arr_loop = [
                    FeatureCategory(asset).apply(niveau),
                    FeatureMathematical(asset, self.lagList, self.monthHorizonList, self.params)
                        .apply(date, niveau, self.idxAssets_start[ticker]+date_idx),
                    FeatureFourierCoeff(asset, self.startDate, self.endDate, self.lagList, self.params)
                        .apply(date, niveau, self.idxAssets_start[ticker]+date_idx),
                    FeatureFinancialData(asset, self.lagList, self.params)
                        .apply(date, niveau, self.idxAssets_start[ticker]+date_idx),
                    FeatureSeasonal(asset, self.startDate, self.endDate, self.lagList, self.params)
                        .apply(date, niveau),
                    FeatureTA(asset, self.startDate, self.endDate, self.lagList, self.params)
                        .apply(date, niveau, self.idxAssets_start[ticker]+date_idx),
                    groupDynDict[ticker]
                ]
                arr[date_idx, tidx, :] = np.concatenate(arr_loop)
                
        # 4) reshape to (nD*nA, nF) and build Polars DataFrame
        arr_flat = arr.reshape(nD * nA, nF)
        dates    = np.repeat(self.business_days.to_numpy(), self.nAssets)
        tickers  = np.tile(list(self.assets.keys()), self.nDates)

        df = pl.DataFrame({
            "date":   dates,
            "ticker": tickers,
            **{feat: arr_flat[:, i] for i, feat in enumerate(flat_features)}
        })
        
        return df
        
    def getTimeFeatures(self) -> tuple[np.array, np.array]:
        # 1) gather feature‑name lists
        exampleAsset = next(iter(self.assets.values()))
        featureNames = [
            FeatureCategory(exampleAsset).getTimeFeatureNames(),
            FeatureMathematical(exampleAsset, self.lagList, self.monthHorizonList, self.params).getTimeFeatureNames(),
            FeatureFourierCoeff(exampleAsset, self.startDate, self.endDate, self.lagList, self.params).getTimeFeatureNames(),
            FeatureFinancialData(exampleAsset, self.lagList, self.params).getTimeFeatureNames(),
            FeatureSeasonal(exampleAsset, self.startDate, self.endDate, self.lagList, self.params).getTimeFeatureNames(),
            FeatureTA(exampleAsset, self.startDate, self.endDate, self.lagList, self.params).getTimeFeatureNames(),
            self.FGD.getTimeFeatureNames()
        ]
        # flatten into one long list
        flat_features = list(itertools.chain.from_iterable(featureNames))
        
        # 2) preallocate arrays
        nD, nA, nT, nF = self.nDates, self.nAssets, self.timesteps, len(flat_features)
        arr     = np.empty((nD, nA, nT, nF), dtype=np.float32)
        metaarr = np.empty((nD, nA, nT, 2), dtype=object)  # [ticker,date]
        
        # 3) fill arrays
        niveau = 1.0
        for date_idx, date in enumerate(self.business_days):
            groupDynDict = self.FGD.apply(date, self.idxAssets_start[ticker]+date_idx)
            for tidx, ticker in enumerate(self.assets.keys()):
                asset = self.assets[ticker]
                metaarr[date_idx, tidx, 0] = ticker
                metaarr[date_idx, tidx, 1] = date

                arr_loop = [
                    FeatureCategory(asset).apply_timeseries(niveau),
                    FeatureMathematical(asset, self.lagList, self.monthHorizonList, self.params)
                        .apply_timeseries(date, niveau, self.idxAssets_start[ticker] + date_idx),
                    FeatureFourierCoeff(asset, self.startDate, self.endDate, self.lagList, self.params)
                        .apply_timeseries(date, niveau, self.idxAssets_start[ticker] + date_idx),
                    FeatureFinancialData(asset, self.lagList, self.params)
                        .apply_timeseries(date, niveau, self.idxAssets_start[ticker] + date_idx),
                    FeatureSeasonal(asset, self.startDate, self.endDate, self.lagList, self.params)
                        .apply_timeseries(date, niveau),
                    FeatureTA(asset, self.startDate, self.endDate, self.lagList, self.params)
                        .apply_timeseries(date, niveau, self.idxAssets_start[ticker] + date_idx),
                    groupDynDict[ticker]
                ]
                block = np.vstack(arr_loop)        # → (nF, nT)
                arr[date_idx, tidx, :, :] = np.transpose(block)  # → (nT, nF)
                
        metaarr = metaarr.reshape(nD * nA, 2)
        arr = arr.reshape(nD * nA, nT, nF)
        
        return metaarr, arr
            
            