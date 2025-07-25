import numpy as np
import pandas as pd
import datetime
from typing import List

from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps
from src.mathTools.TAIndicators import TAIndicators

class FeatureTA():
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'timesteps': 10,
        'lagList': [1, 2, 5, 10, 20, 50, 100, 200, 300, 500],
    }
    
    def __init__(self, 
            asset: AssetDataPolars, 
            startDate: datetime.date, 
            endDate: datetime.date, 
            params: dict = None
        ):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.timesteps = self.params['timesteps']
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.lagList = self.params['lagList']
        
        self.asset = asset
        self.startDate = startDate
        self.endDate = endDate
        
        self.buffer = 21*12+10  # 12 months + 10 days (see also rolling buffer in TAIndicators)
        self.startIdx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(self.startDate) - max(self.lagList, default=0) - self.buffer
        self.endIdx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(self.endDate)
        
        if self.startIdx < 0:
            raise ValueError("Start Date is too old or lag too long.")
        
        self.taindic = TAIndicators(asset.shareprice.slice(self.startIdx, self.endIdx - self.startIdx + 1))
        self.ColumnToUse = self.taindic.getTAColumnNames()
        self.ColumnToUse_timeseries = self.taindic.getTAColumnNames_timeseries()
    
    def getFeatureNames(self) -> list[str]:
        res_raw = [f"FeatureTA_{col}" for col in self.ColumnToUse]
        
        res_lag = []
        for lag in self.lagList:
            for col in self.ColumnToUse:
                res_lag.append(f'FeatureTA_{col}_lag_m{lag}')
        
        return res_raw + res_lag
    
    def getTimeFeatureNames(self) -> list[str]:
        res = [f"FeatureTA_{col}"  for col in self.ColumnToUse_timeseries]
        
        return res
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None):
        if idx is None:
            idx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(date)
            
        if idx  < self.startIdx:
            raise ValueError("Date is too old.")
        
        curClose = self.asset.shareprice['Close'].item(idx)
        curVol = self.asset.shareprice['Volume'].item(idx)
        
        if not curVol >= 2:
            curVol = 1
        
        curTAdf = self.taindic.getReScaledDataFrame(curClose, curVol)
        curTAdf = curTAdf.select(self.ColumnToUse)
        
        niveau = 1.0
        scalingfactor = scaleToNiveau / niveau
        
        idx_adj = idx - self.startIdx
        features = np.array(curTAdf.row(idx_adj))
        
        for lag in self.lagList:
            lagIdx_adj = idx_adj - lag
            features = np.concatenate([features, np.array(curTAdf.row(lagIdx_adj))])
        
        return (features * scalingfactor).astype(np.float32) 
    
    def apply_timeseries(self, date: datetime.date, idx: int = None):
        if idx is None:
            idx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(date)
        if idx  < self.startIdx:
            raise ValueError("Date is too old.")
        
        coreLen = len(self.ColumnToUse_timeseries)
        featuresMat = np.zeros((self.timesteps, coreLen))
        
        curClose = self.asset.shareprice['Close'].item(idx)
        curVol = self.asset.shareprice['Volume'].item(idx)
        
        if not curVol >= 2:
            curVol = 1
        
        curTAdf = self.taindic.getReScaledDataFrame_timeseries(curClose, curVol)
        curTAdf = curTAdf.select(self.ColumnToUse_timeseries)
        
        idx_adj = idx - self.startIdx
        featuresMat = np.array(curTAdf.slice(idx_adj-self.timesteps+1, self.timesteps))
        
        return featuresMat.astype(np.float32) 