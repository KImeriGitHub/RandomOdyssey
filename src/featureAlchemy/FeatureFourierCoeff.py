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
    }
    
    def __init__(self, 
            asset: AssetDataPolars, 
            startDate: pd.Timestamp, 
            endDate:pd.Timestamp, 
            lagList: List[int] = [], 
            monthHorizonList: List[int] = [],
            params: dict = None
        ):
        self.startDate = startDate
        self.endDate = endDate
        self.asset = asset
        self.lagList = lagList
        self.monthHorizonList = monthHorizonList
        
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']

        if self.monthHorizonList == [] and isinstance(self.params['monthsHorizon'], (int, float)):
            raise ValueError("Deprecation warning: monthsHorizonList should be provided as a list. 'monthsHorizon' is deprecated.")
        
        self.buffer = 21*12+10
        self.startIdx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(self.startDate) - max(self.lagList, default=0)-self.buffer
        self.endIdx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(self.endDate)
        
        assert self.startIdx >= 0 + np.max(self.monthHorizonList) * self.idxLengthOneMonth + self.buffer, "Start index is negative."
        
        n_mhl = len(self.monthHorizonList)
        self.PricesPreMatrix_rsme = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.PricesPreMatrix_rsmeRatio = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.PricesPreMatrix_ampcoeff = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.PricesPreMatrix_signcoeff = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.PricesPreMatrix_phasecoeff = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))

        self.ReturnPreMatrix_rsme = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.ReturnPreMatrix_rsmeRatio = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.ReturnPreMatrix_ampcoeff = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.ReturnPreMatrix_signcoeff = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.ReturnPreMatrix_phasecoeff = np.zeros((self.asset.adjClosePrice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.__preprocess_fourierConst()
        

    def __preprocess_fourierConst(self):
        startIdx = self.startIdx
        endIdx = self.endIdx
        
        for m_idx, m_val in enumerate(self.monthHorizonList):
            for idx in range(startIdx, endIdx+1):
                pricesExt = self.asset.adjClosePrice['AdjClose'].slice(
                    idx - m_val*self.idxLengthOneMonth - 1, 
                    m_val * self.idxLengthOneMonth + 2).to_numpy()
                prices = pricesExt[1:]
                returns = pricesExt[1:] / pricesExt[:-1]
                returns_log = np.log((np.clip(returns, 1e-5, 1e5)))

                #Prices
                _, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff(prices, self.multFactor, self.fouriercutoff)
                _, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, prices)

                res_cos = np.array(res_cos)
                res_sin = np.array(res_sin)
                res_amp = np.sqrt(res_cos**2+ res_sin**2)
                res_phase = np.arctan2(res_sin, np.abs(res_cos))
                res_sign = (np.sign(res_cos)+1.0)/2.0

                self.PricesPreMatrix_rsme[idx,0:(self.fouriercutoff),m_idx]         = rsme[:-1]
                self.PricesPreMatrix_rsmeRatio[idx,0:(self.fouriercutoff),m_idx]    = rsme[1:]/rsme[:-1]
                self.PricesPreMatrix_ampcoeff[idx,0:(self.fouriercutoff),m_idx]     = res_amp[1:]
                self.PricesPreMatrix_signcoeff[idx,0:(self.fouriercutoff),m_idx]    = res_sign[1:]
                self.PricesPreMatrix_phasecoeff[idx,0:(self.fouriercutoff),m_idx]   = res_phase[1:]

                #Returns log
                _, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff([0.0] + returns_log + [0.0], self.multFactor, self.fouriercutoff)
                _, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, [0.0] + returns_log + [0.0])

                res_cos = np.array(res_cos)
                res_sin = np.array(res_sin)
                res_amp = np.sqrt(res_cos**2+ res_sin**2)
                res_phase = np.arctan2(res_sin, np.abs(res_cos))
                res_sign = (np.sign(res_cos)+1.0)/2.0

                self.ReturnPreMatrix_rsme[idx,0:(self.fouriercutoff),m_idx]         = rsme[:-1]
                self.ReturnPreMatrix_rsmeRatio[idx,0:(self.fouriercutoff),m_idx]    = rsme[1:]/rsme[:-1]
                self.ReturnPreMatrix_ampcoeff[idx,0:(self.fouriercutoff),m_idx]     = res_amp[1:]
                self.ReturnPreMatrix_signcoeff[idx,0:(self.fouriercutoff),m_idx]    = res_sign[1:]
                self.ReturnPreMatrix_phasecoeff[idx,0:(self.fouriercutoff),m_idx]   = res_phase[1:]
    
    def getFeatureNames(self) -> list[str]:
        res_names = []
        for m in self.monthHorizonList:
            res_names += (
                  [f"Fourier_Price_SignCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)]
                + [f"Fourier_Price_PhaseCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)]
                + [f"Fourier_Price_RSMERatioCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)])
            
            res_names += (
                  [f"Fourier_ReturnLog_SignCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)]
                + [f"Fourier_ReturnLog_PhaseCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)]
                + [f"Fourier_ReturnLog_RSMERatioCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)])

            res_names += (
                  [f"Fourier_Price_RSMECoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)] 
                + [f"Fourier_Price_AmpCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)])

            res_names += (
                  [f"Fourier_ReturnLog_RSMECoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)] 
                + [f"Fourier_ReturnLog_AmpCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)])

            for lag in self.lagList:
                res_names += (
                      [f"Fourier_Price_RSMECoeff_{i}_MH_{m}_lag_m{lag}" for i in range(1,self.fouriercutoff)] 
                    + [f"Fourier_Price_AmpCoeff_{i}_MH_{m}_lag_m{lag}" for i in range(1,self.fouriercutoff)])
                res_names += (
                      [f"Fourier_ReturnLog_RSMECoeff_{i}_MH_{m}_lag_m{lag}" for i in range(1,self.fouriercutoff)]
                    + [f"Fourier_ReturnLog_AmpCoeff_{i}_MH_{m}_lag_m{lag}" for i in range(1,self.fouriercutoff)])
            
        return res_names
    
    def getTimeFeatureNames(self) -> list[str]:
        #todo
        pass
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(date)
        
        niveau = self.asset.adjClosePrice['AdjClose'].item(idx)
        scalingfactor = scaleToNiveau/niveau
        
        features = []
        for m_idx, _ in enumerate(self.monthHorizonList):
            features.extend(self.PricesPreMatrix_signcoeff[idx,:,m_idx])
            features.extend((self.PricesPreMatrix_phasecoeff[idx,:,m_idx] * 2/np.pi)**10) #The values are too close to +- 1
            features.extend(self.PricesPreMatrix_rsmeRatio[idx,:,m_idx])
            
            features.extend(self.ReturnPreMatrix_signcoeff[idx,:,m_idx])
            features.extend((self.ReturnPreMatrix_phasecoeff[idx,:,m_idx] * 2/np.pi)**10) #The values are too close to +- 1
            features.extend(self.ReturnPreMatrix_rsmeRatio[idx,:,m_idx])

            features.extend(self.PricesPreMatrix_rsme[idx,:,m_idx] * scalingfactor)
            features.extend(self.PricesPreMatrix_ampcoeff[idx,:,m_idx] * scalingfactor)

            features.extend(self.ReturnPreMatrix_rsme[idx,:,m_idx] * scaleToNiveau)
            features.extend(self.ReturnPreMatrix_ampcoeff[idx,:,m_idx] * scaleToNiveau)

            for lag in self.lagList:
                idx_lag = idx - lag

                features.extend(self.PricesPreMatrix_rsme[idx_lag,:,m_idx] * scalingfactor)
                features.extend(self.PricesPreMatrix_ampcoeff[idx_lag,:,m_idx] * scalingfactor)

                features.extend(self.ReturnPreMatrix_rsme[idx_lag,:,m_idx] * scaleToNiveau)
                features.extend(self.ReturnPreMatrix_ampcoeff[idx_lag,:,m_idx] * scaleToNiveau)

        return features
    
    def apply_timeseries(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        #todo
        pass