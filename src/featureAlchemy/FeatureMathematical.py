import numpy as np
import pandas as pd
import polars as pl
import datetime
from typing import List

from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureMathematical():
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'timesteps': 10,
    }

    def __init__(self, 
            asset: AssetDataPolars, 
            lagList: List[int] = [],
            monthHorizonList: List[int] = [],
            params: dict = None):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        
        self.asset = asset
        self.lagList = lagList
        self.monthHorizonList = monthHorizonList

        self.timeseries_ivalList = [3,7]

        self.timesteps = self.params['timesteps']
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        
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
        featureNames = [
            'MathFeature_TradedPrice_log',
            "MathFeature_Price_Diff",
            "MathFeature_Price_DiffDiff",
            "MathFeature_Price_logDiff",
            "MathFeature_Price_logDiffDiff",
            "MathFeature_Return",
            "MathFeature_Return_log",
            "MathFeature_PriceAdjustment",
        ]

        for m in self.monthHorizonList:
            featureNames.extend([
                f"MathFeature_Drawdown_MH{m}",
                f"MathFeature_Drawup_MH{m}",
            ])
        
        for lag in self.lagList:
            featureNames.extend([
                f"MathFeature_Price_Diff_lag_m{lag}",
                f"MathFeature_Price_DiffDiff_lag_m{lag}",
                f"MathFeature_Price_logDiff_lag_m{lag}",
                f"MathFeature_Price_logDiffDiff_lag_m{lag}",
                f"MathFeature_Return_lag_m{lag}",
                f"MathFeature_Return_log_lag_m{lag}",
                f"MathFeature_PriceAdjustment_lag_m{lag}",
            ])

            for m in self.monthHorizonList:
                featureNames.extend([
                    f"MathFeature_Drawdown_lag_m{lag}_MH{m}",
                    f"MathFeature_Drawup_lag_m{lag}_MH{m}",
                ])
        return featureNames
    
    def getTimeFeatureNames(self) -> list[str]:
        featureNames = ['MathFeature_TradedPrice']
        for i in range(len(self.timeseries_ivalList)):
            featureNames.append(f"MathFeature_TradedPrice_sp{i}")

        featureNames.extend([
            "MathFeature_Return",
            "MathFeature_PriceAdjustment"
        ])

        return featureNames
    
    def apply(self, date: datetime.date, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(date)
        if idx-max(self.lagList, default=0) < 0 + 4:
            raise ValueError("Lag is too far back.")
        
        niveau = self.prices.item(idx)
        scalingfactor = scaleToNiveau/niveau
        
        mathFeatures = []  # Todo: make immutable
        mathFeatures.append(self.tradedPrice_log.item(idx))
        mathFeatures.extend([
            (self.prices_Diff.item(idx)) * scaleToNiveau, 
            (self.prices_DiffDiff.item(idx)) * scaleToNiveau, 
            (self.prices_logDiff.item(idx)) * scaleToNiveau, 
            (self.prices_logDiffDiff.item(idx)) * scaleToNiveau, 
            (self.pricesReturns.item(idx) * niveau) * scalingfactor, 
            (self.pricesReturns_log.item(idx) * niveau) * scalingfactor, 
            (self.priceAdjustments.item(idx) * niveau) * scalingfactor,
        ])

        for i, _ in enumerate(self.monthHorizonList):
            mathFeatures.extend([
                (self.drawdown[i].item(idx) * niveau) * scalingfactor,
                (self.drawup[i].item(idx) * niveau) * scalingfactor,
            ])
        
        for lag in self.lagList:
            idx_lag = idx - lag
            mathFeatures.extend([
                (self.prices_Diff.item(idx_lag)) * scaleToNiveau, 
                (self.prices_DiffDiff.item(idx_lag)) * scaleToNiveau, 
                (self.prices_logDiff.item(idx_lag)) * scaleToNiveau, 
                (self.prices_logDiffDiff.item(idx_lag)) * scaleToNiveau, 
                (self.pricesReturns.item(idx_lag) * niveau) * scalingfactor, 
                (self.pricesReturns_log.item(idx_lag) * niveau) * scalingfactor,
                (self.priceAdjustments.item(idx_lag) * niveau) * scalingfactor,
            ])

            for i, m in enumerate(self.monthHorizonList):
                mathFeatures.extend([
                    (self.drawdown[i].item(idx_lag) * niveau) * scalingfactor,
                    (self.drawup[i].item(idx_lag) * niveau) * scalingfactor,
                ])
        
        features = np.array(mathFeatures)
        
        return features.astype(np.float32)
    
    def apply_timeseries(self, date: datetime.date, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(date)
        if idx-max(self.lagList, default=0) < 0 + 4:
            raise ValueError("Lag is too far back.")
        if idx - self.timesteps * np.max(self.timeseries_ivalList) < 0:
            raise ValueError("Not enough data for time series.")
        
        coreLen = len(self.getTimeFeatureNames())
        featuresMat = np.zeros((self.timesteps, coreLen))

        adjFactor = self.tradedPrice.item(idx) / self.prices.item(idx) # to avoid leakage
        niveau = self.tradedPrice.item(idx)
        for ts in range(0, self.timesteps):
            idx_ts = idx - (self.timesteps - 1) + ts

            featuresMat[ts, 0] = np.tanh(self.prices.item(idx_ts) * adjFactor / niveau - 1.0)/2.0 + 0.5
            for i, sp in enumerate(self.timeseries_ivalList):
                idx_ts_sp = idx - ((self.timesteps - 1) - ts) * sp
                featuresMat[ts, i+1] = np.tanh(self.prices.item(idx_ts_sp) * adjFactor / niveau - 1.0)/2.0 + 0.5

            featuresMat[ts, coreLen-2] = np.tanh(self.pricesReturns.item(idx_ts)-1.0)/2.0+0.5
            featuresMat[ts, coreLen-1] = np.tanh(self.priceAdjustments.item(idx_ts))/2.0+0.5

        return featuresMat.astype(np.float32)