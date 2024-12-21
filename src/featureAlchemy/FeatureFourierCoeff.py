import numpy as np
import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

class FeatureFourierCoeff():
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
        
        self.PricesPreMatrix = np.zeros(self.asset.adjClosePrice['AdjClose'].len(), 1 + (self.fouriercutoff-1) + (self.fouriercutoff-1))
        self.ReturnPreMatrix = np.zeros(self.asset.adjClosePrice['AdjClose'].len(), 1 + (self.fouriercutoff-1) + (self.fouriercutoff-1))
        self.__preprocess_fourierConst(asset)
        

    def __preprocess_fourierConst(self):
        startIdx = DPl(self.asset.adjClosePrice).getNextLowerIndex(self.startDate)+1
        endIdx = DPl(self.asset.adjClosePrice).getNextLowerIndex(self.endDate)+1
        
        m = self.monthsHorizon
        for idx in range(startIdx, endIdx+1):
            pricesExt = self.asset.adjClosePrice['AdjClose'].slice(idx - m*self.idxLengthOneMonth - 1, m * self.idxLengthOneMonth + 2).to_numpy()
            prices = pricesExt[1:]
            returns = pricesExt[1:] / pricesExt[:-1]
            returns = np.clip(returns, 1e-5, 1e5)
            
            #Prices
            _, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff(prices, self.multFactor, self.fouriercutoff)
            _, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, prices)
            
            res_cos = np.array(res_cos)
            res_sin = np.array(res_sin)
            res_abs = np.sqrt(res_cos**2+ res_sin**2)
            res_sign = (np.sign(res_cos)+1.0)/2.0
            
            self.PricesPreMatrix[idx,0:(2*self.fouriercutoff-1)] = np.concatenate(([rsme], res_abs[1:], res_sign[1:]))
            
            #Returns
            _, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff([1.0] + returns + [1.0], self.multFactor, self.fouriercutoff)
            _, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, [1.0] + returns + [1.0])
            
            res_cos = np.array(res_cos)
            res_sin = np.array(res_sin)
            res_abs = np.sqrt(res_cos**2+ res_sin**2)
            res_sign = (np.sign(res_cos)+1.0)/2.0
            
            self.ReturnPreMatrix[idx,0:(2*self.fouriercutoff-1)] = np.concatenate(([rsme], res_abs[1:], res_sign[1:]))
    
    def getFeatureNames(self) -> list[str]:
        pricesnames = ["Fourier_Price_RSME"] \
            + ["Fourier_Price_AbsCoeff_"+i for i in range(1,self.fouriercutoff)] \
            + ["Fourier_Price_SignCoeff_"+i for i in range(1,self.fouriercutoff)]
        
        returnnames = ["Fourier_Return_RSME"] \
            + ["Fourier_Return_AbsCoeff_"+i for i in range(1,self.fouriercutoff)] \
            + ["Fourier_Return_SignCoeff_"+i for i in range(1,self.fouriercutoff)]
            
        return pricesnames + returnnames
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = -1):
        if idx<0:
            idx = DPl(self.asset.adjClosePrice).getNextLowerIndex(date)+1
        
        niveau = self.asset.adjClosePrice['AdjClose'].item(idx)
        scalingfactor = scaleToNiveau/niveau
        
        features = np.concatenate((self.PricesPreMatrix[idx,:] *scalingfactor, self.ReturnPreMatrix[idx,:] * scalingfactor))
        return features