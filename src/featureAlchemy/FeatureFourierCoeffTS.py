import numpy as np
import datetime
from typing import List, Dict
from numpy.lib.stride_tricks import sliding_window_view

from src.featureAlchemy.IFeature import IFeature
from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureFourierCoeffTS(IFeature):
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 3,
        'multFactor': 8,
        'timesteps': 10,
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
        self.timesteps = self.params['timesteps']
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
            self.PricesPreMatrix_rsme   [idxs, sl, m_idx] =    rsme_p[:, :-1]
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
        
        MH_val = np.min(self.monthHorizonList)
        
        res_names += [f"Fourier_Price_RSMECoeff_{i}_MH_{MH_val}" for i in range(1,self.fouriercutoff)]
        res_names += [f"Fourier_Price_RSMERatioCoeff_{i}_MH_{MH_val}" for i in range(1,self.fouriercutoff)]
        
        return res_names
    
    def apply(self, dates: List[datetime.date]) -> np.ndarray:
        # 1) get integer indices for each date
        idcs = np.array(DOps(self.asset.shareprice)
            .getNextLowerOrEqualIndices(dates), dtype=np.int64)

        # 2) figure out which horizon to use
        MH_val = min(self.monthHorizonList)
        MH_idx = self.monthHorizonList.index(MH_val)

        # 3) build a (n_dates, timesteps) array of time‐step indices
        #    for each date: idx - (timesteps-1) + ts
        ts = np.arange(self.timesteps, dtype=np.int64)
        offsets = ts - (self.timesteps - 1)
        time_idx = idcs[:, None] + offsets[None, :]    # shape (n_dates, timesteps)

        # 4) pull out the raw matrices in one shot
        #    PricesPreMatrix_rsme has shape (T, n_assets, n_horizons)
        raw_rsme  = self.PricesPreMatrix_rsme [time_idx, :, MH_idx]  # → (n_dates, timesteps, n_assets)
        raw_ratio = self.PricesPreMatrix_rsmeRatio[time_idx, :, MH_idx]  # same shape

        # 5) do all the nonlinear transforms at once
        feat1 = np.tanh(raw_rsme)                                   # (n_dates, timesteps, n_assets)
        feat2 = np.tanh(raw_ratio - 1.0) / 2.0 + 0.5                # same shape

        # 6) concatenate into your feature‐matrix and return
        features = np.concatenate([feat1, feat2], axis=2)           # (n_dates, timesteps, 2*n_assets)
        return features.astype(np.float32)