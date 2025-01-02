import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

class FeatureFourierCoeff():
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 10,
        'multFactor': 8,
        'monthsHorizon': 12,
    }
    
    def __init__(self, asset: AssetDataPolars, startDate: pd.Timestamp, endDate:pd.Timestamp, lagList: List[int] = [], params: dict = None):
        self.startDate = startDate
        self.endDate = endDate
        self.asset = asset
        self.lagList = lagList
        
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.monthsHorizon = self.params['monthsHorizon']
        
        self.buffer = 21*12+10
        self.startIdx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(self.startDate) - max(self.lagList, default=0)-self.buffer
        self.endIdx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(self.endDate)
        
        assert self.startIdx >= 0 + self.monthsHorizon * self.idxLengthOneMonth + self.buffer, "Start index is negative."
        
        self.PricesPreMatrix = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), 1 + (self.fouriercutoff-1) + (self.fouriercutoff-1)))
        self.ReturnPreMatrix = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), 1 + (self.fouriercutoff-1) + (self.fouriercutoff-1)))
        self.__preprocess_fourierConst()
        

    def __preprocess_fourierConst(self):
        startIdx = self.startIdx
        endIdx = self.endIdx
        
        m = self.monthsHorizon
        for idx in range(startIdx, endIdx+1):
            pricesExt = self.asset.adjClosePrice['AdjClose'].slice(idx - m*self.idxLengthOneMonth - 1, m * self.idxLengthOneMonth + 2).to_numpy()
            prices = pricesExt[1:]
            returns = pricesExt[1:] / pricesExt[:-1]
            returns_log = np.log((np.clip(returns, 1e-5, 1e5)))
            
            #Prices
            _, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff(prices, self.multFactor, self.fouriercutoff)
            _, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, prices)
            
            res_cos = np.array(res_cos)
            res_sin = np.array(res_sin)
            res_abs = np.sqrt(res_cos**2+ res_sin**2)
            res_sign = (np.sign(res_cos)+1.0)/2.0
            
            self.PricesPreMatrix[idx,0:(2*self.fouriercutoff-1)] = np.concatenate(([rsme], res_abs[1:], res_sign[1:]))
            
            #Returns log
            _, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff([0.0] + returns_log + [0.0], self.multFactor, self.fouriercutoff)
            _, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, [0.0] + returns_log + [0.0])
            
            res_cos = np.array(res_cos)
            res_sin = np.array(res_sin)
            res_abs = np.sqrt(res_cos**2+ res_sin**2)
            res_sign = (np.sign(res_cos)+1.0)/2.0
            
            self.ReturnPreMatrix[idx,0:(2*self.fouriercutoff-1)] = np.concatenate(([rsme], res_abs[1:], res_sign[1:]))
    
    def getFeatureNames(self) -> list[str]:
        pricesnames = ["Fourier_Price_RSME"] \
            + [f"Fourier_Price_AbsCoeff_{i}" for i in range(1,self.fouriercutoff)] \
            + [f"Fourier_Price_SignCoeff_{i}" for i in range(1,self.fouriercutoff)]
        
        returnnames = ["Fourier_ReturnLog_RSME"] \
            + [f"Fourier_ReturnLog_AbsCoeff_{i}" for i in range(1,self.fouriercutoff)] \
            + [f"Fourier_ReturnLog_SignCoeff_{i}" for i in range(1,self.fouriercutoff)]
            
        pricesnames_lag = []
        returnnames_lag = []
        for lag in self.lagList:
            pricesnames_lag += [f"Fourier_Price_lag_m{lag}_RSME"] \
                + [f"Fourier_Price_lag_m{lag}_AbsCoeff_{i}" for i in range(1,self.fouriercutoff)] \
                + [f"Fourier_Price_lag_m{lag}_SignCoeff_{i}" for i in range(1,self.fouriercutoff)]
            returnnames_lag += [f"Fourier_ReturnLog_lag_m{lag}_RSME"] \
                + [f"Fourier_ReturnLog_lag_m{lag}_AbsCoeff_{i}" for i in range(1,self.fouriercutoff)] \
                + [f"Fourier_ReturnLog_lag_m{lag}_SignCoeff_{i}" for i in range(1,self.fouriercutoff)]
            
        return pricesnames + returnnames+ pricesnames_lag + returnnames_lag
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(date)
        
        niveau = self.asset.adjClosePrice['AdjClose'].item(idx)
        scalingfactor = scaleToNiveau/niveau
        
        features = []
        features.append(self.PricesPreMatrix[idx,:] * scalingfactor)
        features.append(self.ReturnPreMatrix[idx,:] * scaleToNiveau)
        
        for lag in self.lagList:
            idx_lag = idx - lag
            
            features.append(self.PricesPreMatrix[idx_lag,:] * scalingfactor)
            features.append(self.ReturnPreMatrix[idx_lag,:] * scaleToNiveau)

        features = np.concatenate(features)
        return features