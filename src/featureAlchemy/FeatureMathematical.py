import numpy as np
import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

class FeatureMathematical():
    
    def __init__(self, asset: AssetDataPolars, startDate: pd.Timestamp, endDate:pd.Timestamp, params: dict = None):
        self.asset = asset
        
        #preprocess
        self.adjPrices = self.asset.adjClosePrice["AdjClose"]
        
        self.prices = self.asset.shareprice["Close"]
        self.prices_Diff = self.prices.diff()
        self.prices_DiffDiff = self.prices_Diff.diff()
        self.prices_log = self.prices.log()
        self.prices_logDiff = self.prices_log.diff()
        self.prices_logDiffDiff = self.prices_logDiff.diff()
        
        self.pricesReturns: pl.Series = (self.prices / self.prices.shift()).clip(min_val=1e-5, max_val=1e2)
        self.pricesReturns_log: pl.Series = (self.pricesReturns.log()+1.0).clip(min_val=1e-5, max_val=1e2)
        
        self.adjRatio = self.adjPrices / self.prices
        
    def getFeatureNames(self) -> list[str]:
        featureNames = [
            "MathFeature_Price_Diff",
            "MathFeature_Price_DiffDiff",
            "MathFeature_Price_logDiff",
            "MathFeature_Price_logDiffDiff",
            "MathFeature_Return",
            "MathFeature_Return_log",
            "MathFeature_Price_Adjustments",
        ]
        return featureNames
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = -1):
        if idx<0:
            idx = DPl(self.asset.adjClosePrice).getNextLowerIndex(date)+1
        
        niveau = self.adjPrices.item(idx)
        
        mathFeatures = np.array([
            self.prices_Diff.item(idx) + niveau, 
            self.prices_DiffDiff.item(idx) + niveau, 
            self.prices_logDiff.item(idx) + niveau, 
            self.prices_logDiffDiff.item(idx) + niveau, 
            self.pricesReturns.item(idx) * niveau, 
            self.pricesReturns_log.item(idx) * niveau, 
            self.adjRatio.item(idx) * niveau, 
        ])
        mathFeatures.clip(1e-5, niveau*10)
        
        scalingfactor = scaleToNiveau/niveau
        
        features = mathFeatures * scalingfactor
        
        return features