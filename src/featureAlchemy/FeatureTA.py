import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List

from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.mathTools.TAIndicators import TAIndicators

class FeatureTA():
    
    def __init__(self, asset: AssetDataPolars, startDate: pd.Timestamp, endDate:pd.Timestamp, lagList: List[int] = []):
        self.asset = asset
        self.startDate = startDate
        self.endDate = endDate
        self.lagList = lagList
        
        self.buffer = 21*12+10  # 12 months + 10 days (see also rolling buffer in TAIndicators)
        self.startIdx = DPl(self.asset.adjClosePrice).getNextLowerIndex(self.startDate)+1 - max(self.lagList, default=0) - self.buffer
        self.endIdx = DPl(self.asset.adjClosePrice).getNextLowerIndex(self.endDate)+1
        
        if self.startIdx < 0:
            raise ValueError("Start Date is too old or lag too long.")
        
        self.taindic = TAIndicators(asset.shareprice.slice(self.startIdx, self.endIdx - self.startIdx + 1))
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
            
        if idx  < self.startIdx:
            raise ValueError("Date is too old.")
        
        curClose = self.asset.shareprice['Close'].item(idx)
        curVol = self.asset.shareprice['Volume'].item(idx)
        
        curVol = 1 if ~(curVol >= 1) else curVol
        
        curTAdf = self.taindic.getReScaledDataFrame(curClose, curVol)
        curTAdf = curTAdf.select(self.ColumnToUse)
        
        niveau = 1.0
        scalingfactor = scaleToNiveau / niveau
        
        idx_adj = idx - self.startIdx
        features = np.array(curTAdf.row(idx_adj))
        
        for lag in self.lagList:
            lagIdx_adj = idx_adj - lag
            features = np.concatenate([features, np.array(curTAdf.row(lagIdx_adj))])
        
        return features * scalingfactor