import numpy as np
import pandas as pd
import polars as pl
import datetime
from typing import List

from src.featureAlchemy.IFeature import IFeature
from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureLSTM(IFeature):
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'timesteps': 10,
    }

    def __init__(self, 
            asset: AssetDataPolars,
            startDate: datetime.date = None, 
            endDate: datetime.date = None, 
            params: dict = None
        ):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.asset = asset

        self.timesteps = self.params['timesteps']
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']

        self.volumeHorizon = self.timesteps
        
        #preprocess
        self.prices = self.asset.shareprice["AdjClose"] #Note: Be wary of leakage
        self.trade_prices = self.asset.shareprice["Close"]
        self.adjustment_factors = self.prices/self.trade_prices
        self.adjopen_prices = self.asset.shareprice["Open"] * self.adjustment_factors
        self.adjhigh_prices = self.asset.shareprice["High"] * self.adjustment_factors
        self.adjlow_prices = self.asset.shareprice["Low"] * self.adjustment_factors

        self.volumes = self.asset.shareprice["Volume"]
        
    def getFeatureNames(self) -> list[str]:
        featureNames = [
            "FeatureLSTM_AdjClose",
            "FeatureLSTM_AdjOpen",
            "FeatureLSTM_AdjHigh",
            "FeatureLSTM_AdjLow",
            "FeatureLSTM_Volume",
            "FeatureLSTM_TradedPrice"
        ]

        return featureNames
    
    def apply(self, dates: List[datetime.date]) -> np.ndarray:
        # compute raw indices per ticker
        idcs = DOps(self.asset.shareprice).getNextLowerOrEqualIndices(dates)
        idcs = np.array(idcs)
        if min(idcs) - (self.timesteps + self.volumeHorizon + 1) < 0:
            idcs = idcs[idcs >= (self.timesteps + self.volumeHorizon + 1)]
        
        nD = len(dates)
        nI = len(idcs)
        coreLen = len(self.getFeatureNames())
        T = self.timesteps

        # pull entire series into numpy once
        prices = self.prices.to_numpy()
        volumes = self.volumes.to_numpy()
        opens = self.adjopen_prices.to_numpy()
        highs = self.adjhigh_prices.to_numpy()
        lows = self.adjlow_prices.to_numpy()
        traded_prices = self.trade_prices.to_numpy()
        roll_min_vol = self.volumes.rolling_min(window_size=self.volumeHorizon).to_numpy()
        roll_max_vol = self.volumes.rolling_max(window_size=self.volumeHorizon).to_numpy()

        # per‐date leakage‐safe factors
        factor = 1.0 / prices[idcs]       # (nD,)

        # build a (nD, T) matrix of base‐time indices
        t_off = np.arange(T) - (T-1)
        baseIdx = idcs[:,None] + t_off[None,:]     # shape (nD,T)

        featuresMat = np.empty((nD, T, coreLen), dtype=np.float32)
        featuresMat.fill(np.nan)

        # compute the “price‐level” feature
        price_lvl = np.tanh(prices[baseIdx]*factor[:,None] - 1)/2 + .5
        opens_lvl = np.tanh(opens[baseIdx] *factor[:,None] - 1)/2 + .5
        highs_lvl = np.tanh(highs[baseIdx] *factor[:,None] - 1)/2 + .5
        lows_lvl  = np.tanh(lows[baseIdx]  *factor[:,None] - 1)/2 + .5
        featuresMat[-nI:,:,0] = price_lvl
        featuresMat[-nI:,:,1] = opens_lvl
        featuresMat[-nI:,:,2] = highs_lvl
        featuresMat[-nI:,:,3] = lows_lvl

        # compute volume levels
        vol_lvl = (volumes[baseIdx] - roll_min_vol[baseIdx]) / (roll_max_vol[baseIdx] - roll_min_vol[baseIdx])
        featuresMat[-nI:,:,4] = np.clip(vol_lvl, 0, 1)

        # traded prices log‐per‐ten
        featuresMat[-nI:,:,5] = traded_prices[baseIdx]

        return featuresMat.astype(np.float32)