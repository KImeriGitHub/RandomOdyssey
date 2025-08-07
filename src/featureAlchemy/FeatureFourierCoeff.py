import numpy as np
import datetime
from typing import List
from numpy.lib.stride_tricks import sliding_window_view

from src.featureAlchemy.IFeature import IFeature
from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureFourierCoeff(IFeature):
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 3,
        'multFactor': 8,
        'lagList': [1, 2, 5, 10, 20, 50, 100, 200, 300, 500],
        'monthsHorizonList': [1, 2, 4, 6, 8, 12],
    }
    
    def __init__(self, 
            asset: AssetDataPolars, 
            startDate: datetime.date, 
            endDate: datetime.date, 
            params: dict = None
        ):
        self.startDate = startDate
        self.endDate = endDate
        self.asset = asset
        
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.lagList = self.params['lagList']
        self.monthHorizonList = self.params['monthsHorizonList']
        
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
        adj = self.asset.shareprice['AdjClose'].to_numpy()
        idxs = np.arange(self.startIdx, self.endIdx+1)
        n_idx = len(idxs)

        for m_idx, m_val in enumerate(self.monthHorizonList):
            win_len = m_val*self.idxLengthOneMonth + 2
            # build shape (n_idx, win_len) array of all consecutive windows ending at each idx
            windows = sliding_window_view(adj, window_shape=win_len)[idxs - win_len + 1]

            # prices and log-returns, both shape (n_idx, win_len-1)
            prices      = windows[:, 1:]
            returns     = prices / windows[:, :-1]
            returns_log = np.log(np.clip(returns, 1e-5, 1e5))

            # ——— PRICES ———
            _, res_cos_p, res_sin_p = SeriesExpansion.getFourierInterpCoeff(
                prices, self.multFactor, self.fouriercutoff)
            _, rsme_p               = SeriesExpansion.getFourierInterpFunct(
                res_cos_p, res_sin_p, prices)
            res_cos_p = np.array(res_cos_p)
            res_sin_p = np.array(res_sin_p)
            amp_p     = np.hypot(res_cos_p, res_sin_p)
            sign_p    = (np.sign(res_cos_p) + 1.) / 2.

            # assign into your 3-D pre-matrices in one go:
            sl = slice(None, self.fouriercutoff-1)
            self.PricesPreMatrix_rsme[idxs, sl, m_idx] =       rsme_p[:, :-1]
            self.PricesPreMatrix_rsmeRatio[idxs, sl, m_idx] =  rsme_p[:, 1:] / rsme_p[:, :-1]
            self.PricesPreMatrix_ampcoeff[idxs, sl, m_idx] =   amp_p[:, 1:]
            self.PricesPreMatrix_signcoeff[idxs, sl, m_idx] =  sign_p[:, 1:]

            # ——— LOG-RETURNS ——— (pad with zeros at start/end)
            ext = np.pad(returns_log, ((0,0),(1,1)), constant_values=0)
            _, res_cos_r, res_sin_r = SeriesExpansion.getFourierInterpCoeff(
                ext, self.multFactor, self.fouriercutoff)
            _, rsme_r               = SeriesExpansion.getFourierInterpFunct(
                res_cos_r, res_sin_r, ext)
            res_cos_r = np.array(res_cos_r)
            res_sin_r = np.array(res_sin_r)
            amp_r     = np.hypot(res_cos_r, res_sin_r)
            sign_r    = (np.sign(res_cos_r) + 1.) / 2.

            self.ReturnPreMatrix_rsme[idxs, sl, m_idx] =   rsme_r[:, :-1]
            self.ReturnPreMatrix_rsmeRatio[idxs, sl, m_idx] = rsme_r[:, 1:] / rsme_r[:, :-1]
            self.ReturnPreMatrix_ampcoeff[idxs, sl, m_idx] =  amp_r[:, 1:]
            self.ReturnPreMatrix_signcoeff[idxs, sl, m_idx] = sign_r[:, 1:]

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
        
        
    def apply(self, dates: List[datetime.date]) -> np.ndarray:
        # compute raw indices per ticker
        idcs = DOps(self.asset.shareprice).getNextLowerOrEqualIndices(dates)
        idcs = np.array(idcs)
        D = len(idcs)

        # 2) dims
        F = self.fouriercutoff - 1
        M = len(self.monthHorizonList)
        C = len(self.getFeatureNames()) // (F * M)

        # 3) pre‑allocate
        feats = np.empty((D, F, C, M), dtype=np.float32)

        # 4) niveau & scaling per date
        prices = self.asset.shareprice['AdjClose'].to_numpy()
        niveau = prices[idcs]                   # (D,)
        sf_price = 1.0 / (niveau + 1e-6)        # (D,)
        sf_return = 1.0                         # scalar

        # 5) build lag indices matrix once
        lags = np.array(self.lagList)           # (L,)
        idx_lags = idcs[:, None] - lags[None, :]  # (D, L)
        
        # 6) fill in each horizon
        for m, _ in enumerate(self.monthHorizonList):
            base = 0
            # prices & returns at t
            feats[:, :, base,   m] = self.PricesPreMatrix_signcoeff[idcs, :, m]
            feats[:, :, base+1, m] = self.PricesPreMatrix_rsmeRatio[idcs, :, m]
            feats[:, :, base+2, m] = self.ReturnPreMatrix_signcoeff[idcs, :, m]
            feats[:, :, base+3, m] = self.ReturnPreMatrix_rsmeRatio[idcs, :, m]
            base += 4
            # scaled price features at t
            feats[:, :, base,   m] = self.PricesPreMatrix_rsme[idcs, :, m] * sf_price[:, None]
            feats[:, :, base+1, m] = self.PricesPreMatrix_ampcoeff[idcs, :, m] * sf_price[:, None]
            # scaled return features at t
            feats[:, :, base+2, m] = self.ReturnPreMatrix_rsme[idcs, :, m] * sf_return
            feats[:, :, base+3, m] = self.ReturnPreMatrix_ampcoeff[idcs, :, m] * sf_return
            base += 4
            # lagged features
            L = len(self.lagList)
            for j in range(L):
                i_lag = idx_lags[:, j]
                feats[:, :, base,   m] = self.PricesPreMatrix_rsme[i_lag, :, m] * sf_price[:, None]
                feats[:, :, base+1, m] = self.PricesPreMatrix_ampcoeff[i_lag, :, m] * sf_price[:, None]
                feats[:, :, base+2, m] = self.ReturnPreMatrix_rsme[i_lag, :, m] * sf_return
                feats[:, :, base+3, m] = self.ReturnPreMatrix_ampcoeff[i_lag, :, m] * sf_return
                base += 4

        # 7) flatten per date
        return feats.reshape(D, -1, order='F')