import numpy as np
import pandas as pd
import polars as pl
import datetime
from typing import List

from src.featureAlchemy.IFeature import IFeature
from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureMathematicalTS(IFeature):
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'timesteps': 10,
        'lagList': [1, 2, 5, 10, 20, 50, 100, 200, 300, 500],
        'monthHorizonList': [1, 2, 4, 6, 8, 12],
    }

    def __init__(self, 
            asset: AssetDataPolars,
            startDate: datetime.date = None, 
            endDate: datetime.date = None, 
            params: dict = None
        ):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        
        self.asset = asset

        self.timeseries_ivalList = [3,7]

        self.timesteps = self.params['timesteps']
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.lagList = self.params['lagList']
        self.monthHorizonList = self.params['monthHorizonList']
        
        #preprocess
        self.tradedPrice = self.asset.shareprice["Close"]
        self.tradedPrice_log = self.tradedPrice.log()
        self.prices = self.asset.shareprice["AdjClose"] #Note: Can be negative #Note: Be wary of leakage
        self.pricesElevated = self.prices - self.prices.min() + 1.0
        self.prices_Diff = self.prices.diff()
        self.prices_DiffDiff = self.prices_Diff.diff()
        self.prices_log = self.pricesElevated.log()
        self.prices_logDiff = self.prices_log.diff()
        self.prices_logDiffDiff = self.prices_logDiff.diff()
        
        self.pricesReturns: pl.Series = (self.pricesElevated / self.pricesElevated.shift()).clip(1e-5, 1e2)
        self.pricesReturns_log: pl.Series = (self.pricesReturns.log()+1.0)
        self.priceAdjustments = (self.prices / self.asset.shareprice["Close"]).diff()  # NOTE: without Diff there might be leakage

        # Drawdown from last highest point
        self.prices_cummax = [None] * len(self.monthHorizonList)
        self.drawdown: List[pl.Series] = [None] * len(self.monthHorizonList)
        for i, window_size in enumerate(self.monthHorizonList):
            self.prices_cummax[i] = self.prices.rolling_max(window_size=self.idxLengthOneMonth*window_size)
            self.drawdown[i] = (self.prices - self.prices_cummax[i]) / self.prices_cummax[i]
        
        # Drawup from last lowest point
        self.prices_cummin = [None] * len(self.monthHorizonList)
        self.drawup: List[pl.Series] = [None] * len(self.monthHorizonList)
        for i, window_size in enumerate(self.monthHorizonList):
            self.prices_cummin[i] = self.prices.rolling_min(window_size=self.idxLengthOneMonth*window_size)
            self.drawup[i] = (self.prices - self.prices_cummin[i]) / self.prices_cummin[i]

        #TODO: Autocovariance and autocorrelation
        #TODO: Idea: How far away is the traded price from a number like 100 or 500 or 1000
        # and did it pass that number already recently
    
    def getFeatureNames(self) -> list[str]:
        featureNames = ['MathFeature_TradedPrice']
        for i in range(len(self.timeseries_ivalList)):
            featureNames.append(f"MathFeature_TradedPrice_sp{i}")

        featureNames.extend([
            "MathFeature_Return",
            "MathFeature_PriceAdjustment"
        ])

        return featureNames
    
    def apply(self, dates: List[datetime.date]) -> np.ndarray:
        # compute raw indices per ticker
        idcs = DOps(self.asset.shareprice).getNextLowerOrEqualIndices(dates)
        idcs = np.array(idcs)
        if min(idcs) - max(self.lagList, default=0) < 0 + 4:
            raise ValueError("Lag is too far back.")
        if min(idcs) - self.timesteps * np.max(self.timeseries_ivalList) < 0:
            raise ValueError("Not enough data for time series.")
        
        nD = len(dates)
        coreLen = len(self.getFeatureNames())
        T = self.timesteps

        # pull entire series into numpy once
        prices = self.prices.to_numpy()
        returns = self.pricesReturns.to_numpy()
        adj   = self.priceAdjustments.to_numpy()
        traded= self.tradedPrice.to_numpy()

        # per‐date leakage‐safe factors
        nivel = traded[idcs]              # (nD,)
        factor= traded[idcs] / prices[idcs]

        # build a (nD, T) matrix of base‐time indices
        t_off = np.arange(T) - (T-1)
        baseIdx = idcs[:,None] + t_off[None,:]     # shape (nD,T)

        # compute the “price‐level” feature
        lvl = np.tanh(prices[baseIdx]*factor[:,None]/nivel[:,None] - 1)/2 + .5
        # compute lagged‐interval features
        featuresMat = np.zeros((nD, T, coreLen), dtype=np.float32)
        featuresMat[:,:,0] = lvl
        for i, sp in enumerate(self.timeseries_ivalList, start=1):
            idx_sp = idcs[:,None] + (t_off*sp)[None,:]
            featuresMat[:,:,i] = np.tanh(prices[idx_sp]*factor[:,None]/nivel[:,None] - 1)/2 + .5

        # compute returns‐and‐adjustments
        featuresMat[:,:,coreLen-2] = np.tanh(returns[baseIdx] - 1)/2 + .5
        featuresMat[:,:,coreLen-1] = np.tanh(adj[baseIdx])/2

        return featuresMat.astype(np.float32)