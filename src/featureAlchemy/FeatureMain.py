import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List
import itertools
import datetime
import logging

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.mathTools.TAIndicators import TAIndicators
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

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
        'timesteps': 15,
    }

    def __init__(self, 
                 assets: Dict[str, AssetDataPolars],
                 startDate: datetime.date, 
                 endDate: datetime.date, 
                 lagList: List[int],
                 monthHorizonList: List[int],
                 params: dict = None,
                 logger: logging.Logger = None):
        
        self.assets = assets
        self.startDate = startDate
        self.endDate = endDate
        self.lagList = lagList
        self.monthHorizonList = monthHorizonList
        
        if logger is None:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            self.logger = logging.getLogger(__name__)
        else:   
            self.logger = logger
        
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.timesteps = self.params['timesteps']
        
        # get business days
        self.tickers = list(self.assets.keys())
        self.nAssets = len(self.tickers)
        
        #preprocess
        self.business_days= pd.bdate_range(start=self.startDate, end=self.endDate).date.tolist()  
        self.startBDate = self.business_days[0]
        self.endBDate = self.business_days[-1]
        self.nDates = len(self.business_days)
        
        self.nAssets = len(self.assets)
        
        self.idxAssets = {ticker: DOps(self.assets[ticker].shareprice).getNextLowerOrEqualIndices(self.business_days) for ticker in self.assets.keys()}
        self.idxAssets_exc = {ticker: DOps(self.assets[ticker].shareprice).getIndices(self.business_days) for ticker in self.assets.keys()}
            
        self.FGD = FeatureGroupDynamic(self.assets, self.startBDate, self.endBDate, self.lagList, self.monthHorizonList, self.params)
        self.logger.info(f"  FeatureMain initialized with {self.nAssets} assets and {self.nDates} dates.")
    
    def getTreeFeatures(self) -> tuple[np.array, np.array, list[str]]:
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
        metaarr = np.empty((nD, nA, 3), dtype=object)  # [date,ticker,close]
        
        # 3) fill arrays
        niveau = 1.0
        FGD_tick_date = {date: self.FGD.apply(date, {tic: idxlist[i] for tic, idxlist in self.idxAssets.items()}) for i, date in enumerate(self.business_days)}
        for tidx, ticker in enumerate(self.assets.keys()):
            self.logger.info(f"  Processing ticker {ticker} ({tidx+1}/{self.nAssets})")
            FC = FeatureCategory(self.assets[ticker], self.params)
            FM = FeatureMathematical(self.assets[ticker], self.lagList, self.monthHorizonList, self.params)
            FFC = FeatureFourierCoeff(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.monthHorizonList, self.params)
            FFD = FeatureFinancialData(self.assets[ticker], self.lagList, self.params)
            FS = FeatureSeasonal(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.params)
            FTA = FeatureTA(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.params)
            idcs = self.idxAssets_exc[ticker]
            for date_idx, date in enumerate(self.business_days):
                if idcs[date_idx] is None:
                    continue
                
                metaarr[date_idx, tidx, 0] = date
                metaarr[date_idx, tidx, 1] = ticker
                metaarr[date_idx, tidx, 2] = self.assets[ticker].shareprice["Close"].item(idcs[date_idx])

                arr[date_idx, tidx, :] = np.concatenate([
                    FC.apply(niveau),
                    FM.apply(date, niveau, idcs[date_idx]),
                    FFC.apply(date, niveau, idcs[date_idx]),
                    FFD.apply(date, niveau, idcs[date_idx]),
                    FS.apply(date, niveau),
                    FTA.apply(date, niveau, idcs[date_idx]),
                    FGD_tick_date[date][ticker]
                ])
                
        # 4) reshape to (nD*nA, nF) and build Polars DataFrame
        metaarr  = metaarr.reshape(nD * nA, 3)
        arr = arr.reshape(nD * nA, nF)
        
        return metaarr, arr, flat_features
        
    def getTimeFeatures(self) -> tuple[np.array, np.array, list[str]]:
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
        FGD_tick_date = {date: self.FGD.apply_timeseries(date, {tic: idxlist[i] for tic, idxlist in self.idxAssets.items()}) for i, date in enumerate(self.business_days)}
        for tidx, ticker in enumerate(self.assets.keys()):
            self.logger.info(f"  Processing ticker {ticker} ({tidx+1}/{self.nAssets})")
            FC = FeatureCategory(self.assets[ticker], self.params)
            FM = FeatureMathematical(self.assets[ticker], self.lagList, self.monthHorizonList, self.params)
            FFC = FeatureFourierCoeff(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.monthHorizonList, self.params)
            FFD = FeatureFinancialData(self.assets[ticker], self.lagList, self.params)
            FS = FeatureSeasonal(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.params)
            FTA = FeatureTA(self.assets[ticker], self.startBDate, self.endBDate, self.lagList, self.params)
            idcs = self.idxAssets_exc[ticker]
            for date_idx, date in enumerate(self.business_days):
                if idcs[date_idx] is None:
                    continue
                metaarr[date_idx, tidx, 0] = date
                metaarr[date_idx, tidx, 1] = ticker
                metaarr[date_idx, tidx, 2] = self.assets[ticker].shareprice["Close"].item(idcs[date_idx])

                arr[date_idx, tidx, :, :] = np.hstack([ 
                    #FC.apply_timeseries(date),
                    FM.apply_timeseries(date, idcs[date_idx]),
                    FFC.apply_timeseries(date, idcs[date_idx]),
                    FFD.apply_timeseries(date, idcs[date_idx]),
                    FS.apply_timeseries(date),
                    FTA.apply_timeseries(date, idcs[date_idx]),
                    FGD_tick_date[date][ticker]
                ])    # (nT, nF)
                
        metaarr = metaarr.reshape(nD * nA, 3)
        arr = arr.reshape(nD * nA, nT, nF)
        
        return metaarr, arr, flat_features