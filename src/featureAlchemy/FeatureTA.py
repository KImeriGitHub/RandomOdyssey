import numpy as np
import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.mathTools.TAIndicators import TAIndicators

class FeatureTA():
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 15,
        'multFactor': 8,
        'monthsHorizon': 12,
    }
    
    def __init__(self, asset: AssetDataPolars, startDate: pd.Timestamp, endDate:pd.Timestamp, params: dict = None):
        self.startDate = startDate
        self.endDate = endDate
        self.asset = asset
        
        # Update default parameters with any provided parameters
        self.params = self.DEFAULT_PARAMS
        if params is not None:
            self.params.update(params)

        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.monthsHorizon = self.params['monthsHorizon']
        
        self.taindic = TAIndicators(asset.shareprice)
        self.TA_relativeColumns, self.relColumnNames = self.taindic.get_relativeColumns()
        self.TA_minmaxed, self.minmaxColumnNames = self.taindic.scale_MinMax()  # TODO: possible data leakage
    
    def getFeatureNames(self) -> list[str]:
        return self.relColumnNames + self.minmaxColumnNames
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = -1):
        if idx<0:
            idx = DPl(self.asset.adjClosePrice).getNextLowerIndex(date)+1
        
        taRow_rel = self.TA_relativeColumns.iloc[idx, :].values.to_numpy()
        taRow_minmax = self.TA_minmaxed.iloc[idx, :].values.to_numpy()
        
        niveau = self.TA_relativeColumns['Close'].iloc(idx)
        scalingfactor = scaleToNiveau/niveau
        
        features = np.concatenate((taRow_rel * scalingfactor, taRow_minmax*scaleToNiveau))
        if np.maximum(np.abs(features-scaleToNiveau))>1e5:
            raise NotImplementedError()    #TODO: solve in debug
        
        return features