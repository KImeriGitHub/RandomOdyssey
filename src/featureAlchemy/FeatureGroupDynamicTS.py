from matplotlib import dates
import numpy as np
import pandas as pd
import polars as pl
import datetime
from typing import Dict, List, Optional

from src.featureAlchemy.IFeature import IFeature
from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureGroupDynamicTS(IFeature):
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'timesteps': 10,
        'lagList': [1, 2, 5, 10, 20, 50, 100, 200, 300, 500],
        'monthHorizonList': [1, 2, 4, 6, 8, 12],
    }

    def __init__(self, 
            assetspl: Dict[str, AssetDataPolars], 
            startDate: datetime.date, 
            endDate: datetime.date, 
            params: dict = None
        ):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        
        self.assetspl = assetspl
        self.startDate = startDate
        self.endDate = endDate
        
        self.timesteps = self.params['timesteps']
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.lagList = self.params['lagList']
        self.monthHorizonList = self.params['monthHorizonList']

        
        idx_to_days_factor = 365.0/255.0
        self.buffer = 2*self.idxLengthOneMonth
        self.startRecord = startDate - pd.Timedelta(
            days = int(max(self.lagList, default=0)*idx_to_days_factor
                + max(self.monthHorizonList, default=0) * self.idxLengthOneMonth 
                + self.buffer)
            )

        self.tickers = list(self.assetspl.keys())
        self.nAssets = len(self.tickers)
        
        #preprocess
        self.business_days= pd.bdate_range(start=self.startRecord, end=self.endDate).date.tolist()  
        self.startBDate = self.business_days[0]
        self.endBDate = self.business_days[-1]
        self.nDates = len(self.business_days)
        
        self.idxAssets: Dict[str, List[int]] = {
            ticker: DOps(self.assetspl[ticker].shareprice).getNextLowerOrEqualIndices(self.business_days) 
            for ticker in self.tickers
        }

        self.__check_assets()

        self.__preprocess()

    def __check_assets(self):
        # Check on idxAssets: any assets first index if not in the business days
        if any(-1 in lst for lst in self.idxAssets.values()):
            raise ValueError("Some assets do not have data for the business days. Please check the asset data.")

    def __preprocess(self):
        self.avgVolume = np.zeros((self.nDates))
        self.minVolume = np.zeros((self.nDates))
        self.maxVolume = np.zeros((self.nDates))
        self.avgReturnPct = np.zeros((self.nDates))
        self.minReturnPct = np.zeros((self.nDates))
        self.maxReturnPct = np.zeros((self.nDates))
        self.weightedIndex = np.zeros((self.nDates))
        
        self.weightPerAsset = {}  #TODO: ideally using outstanding shares instead of volume
        window_size = 6*self.idxLengthOneMonth
        for ticker in self.tickers:
            self.weightPerAsset[ticker] = (
                self.assetspl[ticker].shareprice["Volume"]
                .rolling_mean(window_size = window_size)
                .to_numpy()
            )
        sumPerDate = np.zeros((self.nDates))
        for i, _ in enumerate(self.business_days):
            sumPerDate[i] = np.sum([self.weightPerAsset[ticker][self.idxAssets[ticker][i]] for ticker in self.tickers])
        for ticker in self.tickers:
            for i, _ in enumerate(self.business_days):
                assetIdx = self.idxAssets[ticker][i]
                self.weightPerAsset[ticker][assetIdx] = self.weightPerAsset[ticker][assetIdx] / sumPerDate[i]
                
        # weightedIndex is close price times weight and then sum over all assets
        self.weightedIndex = np.zeros((self.nDates))
        for i, _ in enumerate(self.business_days):
            self.weightedIndex[i] = np.sum([self.assetspl[ticker].shareprice["Close"].item(self.idxAssets[ticker][i]) 
                        * self.weightPerAsset[ticker][self.idxAssets[ticker][i]] for ticker in self.tickers])
        
        #Group Dynamics
        self.allReturnsPct = np.zeros((self.nDates, self.nAssets))
        self.allVolumes = np.zeros((self.nDates, self.nAssets))
        for i, _ in enumerate(self.business_days):
            self.allReturnsPct[i] = np.array([self.assetspl[ticker].shareprice["Close"].pct_change().item(self.idxAssets[ticker][i]) for ticker in self.tickers])
            self.allVolumes[i] = np.array([self.assetspl[ticker].shareprice["Volume"].item(self.idxAssets[ticker][i]) for ticker in self.tickers])

            self.minVolume[i] = np.min(self.allVolumes[i])
            self.maxVolume[i] = np.max(self.allVolumes[i])
            self.avgVolume[i] = np.mean(self.allVolumes[i])
            self.minReturnPct[i] = np.min(self.allReturnsPct[i])
            self.maxReturnPct[i] = np.max(self.allReturnsPct[i])
            self.avgReturnPct[i] = np.mean(self.allReturnsPct[i])
            
        self.featureDict_RetGrLvl: Dict[str, np.array] = {}
        self.featureDict_VolGrLvl: Dict[str, np.array] = {}
        self.featureDict_RetGrRk: Dict[str, np.array] = {}
        self.featureDict_VolGrRk: Dict[str, np.array] = {}
        self.featureDict_WeightedIndexPct: Dict[str, np.array] = {}
        for ticker, asset in self.assetspl.items():
            closePct: np.array = asset.shareprice["Close"].pct_change().gather(self.idxAssets[ticker]).to_numpy()
            curVol: np.array = asset.shareprice["Volume"].gather(self.idxAssets[ticker]).to_numpy()
            self.featureDict_RetGrLvl[ticker] = ((closePct-self.minReturnPct) / (self.maxReturnPct-self.minReturnPct))
            self.featureDict_VolGrLvl[ticker] = ((curVol-self.minVolume) / (self.maxVolume-self.minVolume))
            self.featureDict_RetGrRk[ticker] = (np.sum(self.allReturnsPct[:,:] <= closePct[:, None], axis=1) / self.nAssets)
            self.featureDict_VolGrRk[ticker] = (np.sum(self.allVolumes[:,:] <= curVol[:, None], axis=1) / self.nAssets)
        
    def apply(self, dates: List[datetime.date]) -> Dict[str, np.ndarray]:
        # Precompute index dictionaries
        idcs_dict = {
            ticker: DOps(asset.shareprice).getNextLowerOrEqualIndices(dates)
            for ticker, asset in self.assetspl.items()
        }

        nD = len(dates)
        nT = self.timesteps
        nF = len(self.getFeatureNames())
        featureDict: Dict[str, np.ndarray] = {}

        # Precompute timestep offsets
        offsets = np.arange(nT)[None, :]  # shape (1, nT)

        for ticker in self.assetspl:
            # 1D array of adjusted indices per date
            raw_idcs = np.array(idcs_dict[ticker], dtype=int) - self.idxAssets[ticker][0]
            # Build 2D matrix of positions: shape (nD, nT)
            positions = raw_idcs[:, None] - (nT - 1) + offsets

            # Allocate feature array
            feats = np.empty((nD, nT, nF), dtype=np.float32)

            # Shortcut references
            f_ret_lvl   = self.featureDict_RetGrLvl[ticker]
            f_vol_lvl   = self.featureDict_VolGrLvl[ticker]
            f_ret_rk    = self.featureDict_RetGrRk[ticker]
            f_vol_rk    = self.featureDict_VolGrRk[ticker]
            avg_ret     = self.avgReturnPct
            widx        = self.weightedIndex
            M1 = self.idxLengthOneMonth

            # Fill tensorized features
            feats[..., 0] = np.clip(f_ret_lvl[positions], 0.0, 1.0)
            feats[..., 1] = np.clip(f_vol_lvl[positions], 0.0, 1.0)
            feats[..., 2] = np.clip(f_ret_rk[positions], 0.0, 1.0)
            feats[..., 3] = np.clip(f_vol_rk[positions], 0.0, 1.0)
            feats[..., 4] = np.tanh(avg_ret[positions]) / 2.0 + 0.5

            # Percent change from previous timestep
            prev_pos = positions - 1
            feats[..., 5] = np.tanh(widx[positions] / (widx[prev_pos] + 1e-6) - 1.0) / 2.0 + 0.5

            # Month-over-month percent change
            prev_m1 = positions - M1 - 1
            feats[..., 6] = np.tanh(widx[positions] / (widx[prev_m1] + 1e-6) - 1.0) / 2.0 + 0.5

            featureDict[ticker] = feats

        return featureDict
    
    def getFeatureNames(self) -> list[str]:
        featureNames = [
            "FeatureGroup_RetGrLvl",
            "FeatureGroup_VolGrLvl",
            "FeatureGroup_RetGrRk",
            "FeatureGroup_VolGrRk",
            "FeatureGroup_AvgReturnPct",
            "FeatureGroup_WeightedIndexPct",
            "FeatureGroup_WeightedIndexPct_MH_1",
        ]
        
        return featureNames