import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List

from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.mathTools.TAIndicators import TAIndicators

class FeatureTA():
    
    def __init__(self, asset: AssetDataPolars, lagList: List[int] = []):
        self.asset = asset
        self.lagList = lagList
        self.taindic = TAIndicators(asset.shareprice)
        self.ColumnToUse = self.taindic.getTAColumnNames()
    
    def getFeatureNames(self) -> list[str]:
        res_raw = [f"FeatureTA_{col}" for col in self.ColumnToUse]
        
        res_lag = []
        for lag in self.lagList:
            for col in self.ColumnToUse:
                res_lag.append(f'FeatureTA_{col}_lag_m{lag}')
        
        return res_raw + res_lag
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None):
        if idx is None:
            idx = DPl(self.asset.adjClosePrice).getNextLowerIndex(date)+1
            
        if idx - max(self.lagList, default=0) < 0:
            raise ValueError("Not enough data to calculate TA features")
        
        curClose = self.asset.shareprice['Close'].item(idx)
        curVol = self.asset.shareprice['Volume'].item(idx)
        
        curTAdf = self.taindic.getReScaledDataFrame(curClose, curVol)
        curTAdf = curTAdf.select(self.ColumnToUse)
        
        niveau = 1.0
        scalingfactor = scaleToNiveau / niveau
        
        features = np.array(curTAdf.row(idx))
        
        for lag in self.lagList:
            lagIdx = idx - lag
            features = np.concatenate([features, np.array(curTAdf.row(lagIdx))])
        
        return features * scalingfactor