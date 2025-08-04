import numpy as np
import pandas as pd
import datetime
from typing import List

from src.featureAlchemy.IFeature import IFeature
from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps
from src.mathTools.TAIndicators import TAIndicators

class FeatureTATS(IFeature):
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
        #self.ColumnToUse = self.taindic.getTAColumnNames()
        self.ColumnToUse_timeseries = self.taindic.getTAColumnNames_timeseries()
    
    def getFeatureNames(self) -> list[str]:
        res = [f"FeatureTA_{col}"  for col in self.ColumnToUse_timeseries]
        
        return res
    
    def apply(self, dates: List[datetime.date]) -> np.ndarray:
        # compute raw indices per ticker
        idcs = DOps(self.asset.shareprice).getNextLowerOrEqualIndices(dates)
        idcs = np.array(idcs)
        if min(idcs) < self.startIdx:
            raise ValueError("Date is too old.")

        # 2) grab prices & volumes at those indices
        sp = self.asset.shareprice
        closes = sp['Close'].to_numpy()[idcs]
        vols   = sp['Volume'].to_numpy()[idcs]
        vols   = np.where(vols < 2, 1.0, vols)
        
        # 3) build frames
        dfTA_exprs = [self.taindic.getRescaledExprs_timeseries(closes[i], vols[i]) for i in range(len(closes))]
        cur_arrs = [self.taindic.tadata.select(dfTA_expr).to_numpy() for dfTA_expr in dfTA_exprs]

        idxs = idcs - self.startIdx           # shape (N,)

        # 2. View of *every* length‑t window (no extra memory!)
        #    result shape = (n_rows‑t+1, timesteps, n_features)
        windows: list[np.ndarray] = [
            np.lib.stride_tricks.sliding_window_view(
                    cur_arr, (self.timesteps, cur_arr.shape[1]))[:, 0]
            for cur_arr in cur_arrs
        ]
        
        features  = np.array([
            window[idxs[i]] for i, window in enumerate(windows)
        ])

        return features.astype(np.float32) 