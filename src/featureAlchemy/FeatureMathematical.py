import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

class FeatureMathematical():
    
    def __init__(self, asset: AssetDataPolars, lagList: List[int] = []):
        self.asset = asset
        self.lagList = lagList
        
        #preprocess
        self.tradedPrice_log = self.asset.shareprice["Close"].log()
        self.prices = self.asset.adjClosePrice["AdjClose"] #Note: Can be negative #Note: Be wary of leakage
        self.pricesElevated = self.prices - self.prices.min() + 1.0
        self.prices_Diff = self.prices.diff()
        self.prices_DiffDiff = self.prices_Diff.diff()
        self.prices_log = self.pricesElevated.log()
        self.prices_logDiff = self.prices_log.diff()
        self.prices_logDiffDiff = self.prices_logDiff.diff()
        
        self.pricesReturns: pl.Series = (self.pricesElevated / self.pricesElevated.shift()).clip(1e-5, 1e2)
        self.pricesReturns_log: pl.Series = (self.pricesReturns.log()+1.0)
        
        # Drawdown from last highest point
        self.prices_cummax = self.prices.rolling_max(window_size=21*6)
        self.drawdown = (self.prices - self.prices_cummax) / self.prices_cummax
        
        # Drawup from last lowest point
        self.prices_cummin = self.prices.rolling_min(window_size=21*6)
        self.drawup = (self.prices - self.prices_cummin) / self.prices_cummin

        self.priceAdjustments = (self.prices / self.asset.shareprice["Close"]).diff()  # NOTE: without Diff there might be leakage
        
        #TODO: Idea: include 'monthsHorizon' and 'idxLengthOneMonth' in rolling prices.
        #TODO: Idea: How far away is the traded price from a number like 100 or 500 or 1000
        # and did it pass that number already recently
        
    def getFeatureNames(self) -> list[str]:
        featureNames = [
            'MathFeature_TradedPrice_Rank',
            "MathFeature_Price_Diff",
            "MathFeature_Price_DiffDiff",
            "MathFeature_Price_logDiff",
            "MathFeature_Price_logDiffDiff",
            "MathFeature_Return",
            "MathFeature_Return_log",
            "MathFeature_Drawdown",
            "MathFeature_Drawup",
            "MathFeature_PriceAdjustment",
        ]
        
        for lag in self.lagList:
            featureNames.extend([
                f"MathFeature_Price_Diff_lag_m{lag}",
                f"MathFeature_Price_DiffDiff_lag_m{lag}",
                f"MathFeature_Price_logDiff_lag_m{lag}",
                f"MathFeature_Price_logDiffDiff_lag_m{lag}",
                f"MathFeature_Return_lag_m{lag}",
                f"MathFeature_Return_log_lag_m{lag}",
                f"MathFeature_Drawdown_lag_m{lag}",
                f"MathFeature_Drawup_lag_m{lag}",
                f"MathFeature_PriceAdjustment_lag_m{lag}",
            ])
        return featureNames
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DPl(self.asset.adjClosePrice).getNextLowerIndex(date)+1
        if idx-max(self.lagList, default=0) < 0 + 4:
            raise ValueError("Lag is too far back.")
        
        niveau = self.prices.item(idx)
        scalingfactor = scaleToNiveau/niveau
        
        mathFeatures = []  # Todo: make immutable
        mathFeatures.append(self.tradedPrice_log.item(idx))
        mathFeatures.extend([
            (self.prices_Diff.item(idx) + niveau) * scalingfactor, 
            (self.prices_DiffDiff.item(idx) + niveau) * scalingfactor, 
            (self.prices_logDiff.item(idx) + niveau) * scalingfactor, 
            (self.prices_logDiffDiff.item(idx) + niveau) * scalingfactor, 
            (self.pricesReturns.item(idx) * niveau) * scalingfactor, 
            (self.pricesReturns_log.item(idx) * niveau) * scalingfactor, 
            (self.drawdown.item(idx) * niveau) * scalingfactor,
            (self.drawup.item(idx) * niveau) * scalingfactor,
            (self.priceAdjustments.item(idx) * niveau) * scalingfactor,
        ])
        
        for lag in self.lagList:
            idx_lag = idx - lag
            mathFeatures.extend([
                (self.prices_Diff.item(idx_lag) + niveau) * scalingfactor, 
                (self.prices_DiffDiff.item(idx_lag) + niveau) * scalingfactor, 
                (self.prices_logDiff.item(idx_lag) + niveau) * scalingfactor, 
                (self.prices_logDiffDiff.item(idx_lag) + niveau) * scalingfactor, 
                (self.pricesReturns.item(idx_lag) * niveau) * scalingfactor, 
                (self.pricesReturns_log.item(idx_lag) * niveau) * scalingfactor,
                (self.drawdown.item(idx_lag) * niveau) * scalingfactor,
                (self.drawup.item(idx_lag) * niveau) * scalingfactor,
                (self.priceAdjustments.item(idx_lag) * niveau) * scalingfactor,
            ])
        
        features = np.array(mathFeatures)
        
        return features