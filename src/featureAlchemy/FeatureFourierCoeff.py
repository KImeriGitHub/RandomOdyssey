import numpy as np
import datetime
from typing import List

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureFourierCoeff():
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 10,
        'multFactor': 8,
        'timesteps': 10,
    }
    
    def __init__(self, 
            asset: AssetDataPolars, 
            startDate: datetime.date, 
            endDate: datetime.date, 
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
        self.timesteps = self.params['timesteps']

        if self.monthHorizonList == [] and isinstance(self.params['monthsHorizon'], (int, float)):
            raise ValueError("Deprecation warning: monthsHorizonList should be provided as a list. 'monthsHorizon' is deprecated.")
        
        self.buffer = 10
        self.startIdx = (DOps(self.asset.shareprice).getNextLowerOrEqualIndex(self.startDate) 
                - max(self.lagList, default=0) - max(self.monthHorizonList, default=0) * self.idxLengthOneMonth - self.buffer)
        self.endIdx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(self.endDate)
        
        assert self.startIdx >= 0, "Start index is negative."
        
        n_mhl = len(self.monthHorizonList)
        self.PricesPreMatrix_rsme = np.zeros((self.asset.shareprice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.PricesPreMatrix_rsmeRatio = np.zeros((self.asset.shareprice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.PricesPreMatrix_ampcoeff = np.zeros((self.asset.shareprice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.PricesPreMatrix_signcoeff = np.zeros((self.asset.shareprice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))

        self.ReturnPreMatrix_rsme = np.zeros((self.asset.shareprice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.ReturnPreMatrix_rsmeRatio = np.zeros((self.asset.shareprice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.ReturnPreMatrix_ampcoeff = np.zeros((self.asset.shareprice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.ReturnPreMatrix_signcoeff = np.zeros((self.asset.shareprice['AdjClose'].len(), (self.fouriercutoff-1), n_mhl))
        self.__preprocess_fourierConst()
        
    def __preprocess_fourierConst(self):
        startIdx = self.startIdx
        endIdx = self.endIdx
        
        for m_idx, m_val in enumerate(self.monthHorizonList):
            pricesExt_dict = {
                idx: self.asset.shareprice['AdjClose'].slice(
                    idx - m_val*self.idxLengthOneMonth - 1, m_val * self.idxLengthOneMonth + 2
                    ).to_numpy() 
                for idx in range(startIdx, endIdx+1)
            }
            for idx in range(startIdx, endIdx+1):
                pricesExt = pricesExt_dict[idx]
                prices = pricesExt[1:]
                returns = pricesExt[1:] / pricesExt[:-1]
                returns_log = np.log((np.clip(returns, 1e-5, 1e5)))

                #Prices
                _, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff(prices, self.multFactor, self.fouriercutoff)
                _, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, prices)

                res_cos = np.array(res_cos)
                res_sin = np.array(res_sin)
                res_amp = np.sqrt(res_cos**2+ res_sin**2)
                res_sign = (np.sign(res_cos)+1.0)/2.0

                self.PricesPreMatrix_rsme[idx,0:(self.fouriercutoff-1),m_idx]         = rsme[:-1]
                self.PricesPreMatrix_rsmeRatio[idx,0:(self.fouriercutoff-1),m_idx]    = rsme[1:]/rsme[:-1]
                self.PricesPreMatrix_ampcoeff[idx,0:(self.fouriercutoff-1),m_idx]     = res_amp[1:]
                self.PricesPreMatrix_signcoeff[idx,0:(self.fouriercutoff-1),m_idx]    = res_sign[1:]

                #Returns log
                _, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff([0.0] + returns_log + [0.0], self.multFactor, self.fouriercutoff)
                _, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, [0.0] + returns_log + [0.0])

                res_cos = np.array(res_cos)
                res_sin = np.array(res_sin)
                res_amp = np.sqrt(res_cos**2+ res_sin**2)
                res_sign = (np.sign(res_cos)+1.0)/2.0

                self.ReturnPreMatrix_rsme[idx,0:(self.fouriercutoff-1),m_idx]         = rsme[:-1]
                self.ReturnPreMatrix_rsmeRatio[idx,0:(self.fouriercutoff-1),m_idx]    = rsme[1:]/rsme[:-1]
                self.ReturnPreMatrix_ampcoeff[idx,0:(self.fouriercutoff-1),m_idx]     = res_amp[1:]
                self.ReturnPreMatrix_signcoeff[idx,0:(self.fouriercutoff-1),m_idx]    = res_sign[1:]
    
    def getFeatureNames(self) -> list[str]:
        res_names = []
        for m in self.monthHorizonList:
            res_names += (
                  [f"Fourier_Price_SignCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)]
                + [f"Fourier_Price_RSMERatioCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)])
            
            res_names += (
                  [f"Fourier_ReturnLog_SignCoeff_{i}_MH_{m}" for i in range(1,self.fouriercutoff)]
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
        res_names = []
        
        MH_val = np.min(self.monthHorizonList)
        
        res_names += [f"Fourier_Price_RSMECoeff_{i}_MH_{MH_val}" for i in range(1,self.fouriercutoff)]
        res_names += [f"Fourier_Price_RSMERatioCoeff_{i}_MH_{MH_val}" for i in range(1,self.fouriercutoff)]
        
        return res_names
        
        
    def apply(self, date: datetime.date, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(date)
        
        MHL_len = len(self.monthHorizonList)
        coreLen = len(self.getFeatureNames()) // MHL_len // (self.fouriercutoff-1)
        features = np.full((self.fouriercutoff-1, coreLen, MHL_len), np.nan)
        
        niveau = self.asset.shareprice['AdjClose'].item(idx)
        scalingfactor = scaleToNiveau/niveau
        
        for m_idx, _ in enumerate(self.monthHorizonList):
            features[:, 0, m_idx] = self.PricesPreMatrix_signcoeff[idx,:,m_idx]
            features[:, 1, m_idx] = self.PricesPreMatrix_rsmeRatio[idx,:,m_idx]
            
            features[:, 2, m_idx] = self.ReturnPreMatrix_signcoeff[idx,:,m_idx]
            features[:, 3, m_idx] = self.ReturnPreMatrix_rsmeRatio[idx,:,m_idx]

            features[:, 4, m_idx] = self.PricesPreMatrix_rsme[idx,:,m_idx] * scalingfactor
            features[:, 5, m_idx] = self.PricesPreMatrix_ampcoeff[idx,:,m_idx] * scalingfactor

            features[:, 6, m_idx] = self.ReturnPreMatrix_rsme[idx,:,m_idx] * scaleToNiveau
            features[:, 7, m_idx] = self.ReturnPreMatrix_ampcoeff[idx,:,m_idx] * scaleToNiveau

            enuCounter = 8
            for lag in self.lagList:
                idx_lag = idx - lag

                features[:, enuCounter, m_idx] = (self.PricesPreMatrix_rsme[idx_lag,:,m_idx] * scalingfactor)
                features[:, enuCounter + 1, m_idx] = (self.PricesPreMatrix_ampcoeff[idx_lag,:,m_idx] * scalingfactor)

                features[:, enuCounter + 2, m_idx] = (self.ReturnPreMatrix_rsme[idx_lag,:,m_idx] * scaleToNiveau)
                features[:, enuCounter + 3, m_idx] = (self.ReturnPreMatrix_ampcoeff[idx_lag,:,m_idx] * scaleToNiveau)
                
                enuCounter += 4

        return features.flatten('F').astype(np.float32)
    
    def apply_timeseries(self, date: datetime.date, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DOps(self.asset.shareprice).getNextLowerOrEqualIndex(date)
        
        coreLen = len(self.getTimeFeatureNames())
        featuresMat = np.zeros((self.timesteps, coreLen))
        
        MH_val = np.min(self.monthHorizonList)
        MH_val_idx = self.monthHorizonList.index(MH_val)
        
        for ts in range(0, self.timesteps):
            idx_ts = idx - (self.timesteps - 1) + ts
            featuresMat[ts, 0:4] = np.tanh(self.PricesPreMatrix_rsme[idx_ts, :, MH_val_idx])
            featuresMat[ts, 4:8] = np.tanh(self.PricesPreMatrix_rsmeRatio[idx_ts, :, MH_val_idx] - 0.5)
            
        return featuresMat.astype(np.float32)