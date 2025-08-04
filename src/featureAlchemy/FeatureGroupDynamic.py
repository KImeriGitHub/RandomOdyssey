from itertools import chain
from matplotlib import dates
import numpy as np
import pandas as pd
import polars as pl
import datetime
from typing import Dict, List, Optional

from src.featureAlchemy.IFeature import IFeature
from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureGroupDynamic(IFeature):
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
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
        
        self.idxAssets_busi: Dict[str, List[int]] = {
            ticker: DOps(self.assetspl[ticker].shareprice).getNextLowerOrEqualIndices(self.business_days) 
            for ticker in self.tickers
        }

        mappeddates_dict: Dict[str, List[datetime.date]] = {
            ticker: self.assetspl[ticker].shareprice['Date'].gather(self.idxAssets_busi[ticker]).to_list()
            for ticker in self.tickers
        }
        self.trading_days = sorted({d for d in chain.from_iterable(mappeddates_dict.values())})
        self.idxAssets_trad: Dict[str, List[int]] = {
            ticker: DOps(self.assetspl[ticker].shareprice).getNextLowerOrEqualIndices(self.trading_days) 
            for ticker in self.tickers
        }
        self.__check_assets()

        self.__preprocess()

    def __check_assets(self):
        # Check on idxAssets: any assets first index if not in the business days
        if any(-1 in lst for lst in self.idxAssets_busi.values()):
            raise ValueError("Some assets do not have data for the business days. Please check the asset data.")
        if any(-1 in lst for lst in self.idxAssets_trad.values()):
            raise ValueError("Some assets do not have data for the trading days. Please check the asset data.")

    def __preprocess(self):
        nD = len(self.trading_days)
        self.avgVolume = np.zeros((nD))
        self.minVolume = np.zeros((nD))
        self.maxVolume = np.zeros((nD))
        self.avgReturnPct = np.zeros((nD))
        self.minReturnPct = np.zeros((nD))
        self.maxReturnPct = np.zeros((nD))
        self.weightedIndex = np.zeros((nD))

        self.weightPerAsset = {}  #TODO: ideally using outstanding shares instead of volume
        window_size = 6*self.idxLengthOneMonth
        for ticker in self.tickers:
            self.weightPerAsset[ticker] = (
                self.assetspl[ticker].shareprice["Volume"]
                .rolling_mean(window_size = window_size)
                .to_numpy()
            )
        sumPerDate = np.zeros((nD))
        for i, _ in enumerate(self.trading_days):
            sumPerDate[i] = np.sum([self.weightPerAsset[ticker][self.idxAssets_trad[ticker][i]] for ticker in self.tickers])
        for ticker in self.tickers:
            for i, _ in enumerate(self.trading_days):
                assetIdx = self.idxAssets_trad[ticker][i]
                self.weightPerAsset[ticker][assetIdx] = self.weightPerAsset[ticker][assetIdx] / sumPerDate[i]
                
        # weightedIndex is close price times weight and then sum over all assets
        self.weightedIndex = np.zeros((nD))
        for i, _ in enumerate(self.trading_days):
            self.weightedIndex[i] = np.sum([
                self.assetspl[ticker].shareprice["AdjClose"].item(self.idxAssets_trad[ticker][i])  # Might lead to leakage if not using pct_change
                * self.weightPerAsset[ticker][self.idxAssets_trad[ticker][i]] 
            for ticker in self.tickers])
        
        #Group Dynamics
        self.allReturnsPct = np.zeros((nD, self.nAssets))
        self.allVolumes = np.zeros((nD, self.nAssets))
        for i, _ in enumerate(self.trading_days):
            self.allReturnsPct[i] = np.array([self.assetspl[ticker].shareprice["AdjClose"].pct_change().item(self.idxAssets_trad[ticker][i]) for ticker in self.tickers])
            self.allVolumes[i] = np.array([self.assetspl[ticker].shareprice["Volume"].item(self.idxAssets_trad[ticker][i]) for ticker in self.tickers])

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
            closePct: np.array = asset.shareprice["AdjClose"].pct_change().gather(self.idxAssets_trad[ticker]).to_numpy()
            curVol: np.array = asset.shareprice["Volume"].gather(self.idxAssets_trad[ticker]).to_numpy()
            self.featureDict_RetGrLvl[ticker] = ((closePct-self.minReturnPct) / (self.maxReturnPct-self.minReturnPct))
            self.featureDict_VolGrLvl[ticker] = ((curVol-self.minVolume) / (self.maxVolume-self.minVolume))
            self.featureDict_RetGrRk[ticker] = (np.sum(self.allReturnsPct[:,:] <= closePct[:, None], axis=1) / self.nAssets)
            self.featureDict_VolGrRk[ticker] = (np.sum(self.allVolumes[:,:] <= curVol[:, None], axis=1) / self.nAssets)
        
    def apply(self, dates: List[datetime.date]) -> Dict[str, np.ndarray]:
        # compute raw indices per ticker
        idcs_dict = {
            t: DOps(a.shareprice).getNextLowerOrEqualIndices(dates)
            for t, a in self.assetspl.items()
        }
        
        out: Dict[str, np.ndarray] = {}
        n_dates = len(dates)
        L = len(self.lagList)
        M = len(self.monthHorizonList)
        # total features: 4 group + 2 fixed + 6 per lag + (L*M) horizons
        n_feats = 6 + 6*L + L*M

        for t, asset in self.assetspl.items():
            base_idx = np.array(idcs_dict[t]) - self.idxAssets_trad[t][0]
            F = np.empty((n_dates, n_feats), dtype=np.float32)
            c = 0

            # --- current features ---
            F[:, c] = self.featureDict_RetGrLvl[t][base_idx];      c += 1
            F[:, c] = self.featureDict_VolGrLvl[t][base_idx];      c += 1
            F[:, c] = self.featureDict_RetGrRk[t][base_idx];       c += 1
            F[:, c] = self.featureDict_VolGrRk[t][base_idx];       c += 1

            F[:, c] = self.avgReturnPct[base_idx];                 c += 1
            prev = np.maximum(base_idx, 1) - 1
            F[:, c] = self.weightedIndex[base_idx] / (self.weightedIndex[prev] + 1e-6) - 1.0
            c += 1

            # --- lagged features ---
            for lag in self.lagList:
                idx_l = base_idx - lag
                F[:, c] = self.featureDict_RetGrLvl[t][idx_l];     c += 1
                F[:, c] = self.featureDict_VolGrLvl[t][idx_l];     c += 1
                F[:, c] = self.featureDict_RetGrRk[t][idx_l];      c += 1
                F[:, c] = self.featureDict_VolGrRk[t][idx_l];      c += 1
                F[:, c] = self.avgReturnPct[idx_l];                c += 1
                prev = np.maximum(idx_l, 1) - 1
                F[:, c] = self.weightedIndex[idx_l] / (self.weightedIndex[prev] + 1e-6) - 1.0
                c += 1

            # --- lag × horizon features ---
            for lag in self.lagList:
                idx_l = base_idx - lag
                for m in self.monthHorizonList:
                    idx_mh = idx_l - m * self.idxLengthOneMonth
                    prev = np.maximum(idx_mh, 1) - 1
                    F[:, c] = self.weightedIndex[idx_l] / (self.weightedIndex[prev] + 1e-6) - 1.0
                    c += 1

            out[t] = F

        #TODO: Cointegration, autcorrelation, and other features can be added here

        return out
        
    def getFeatureNames(self) -> list[str]:
        featureNames = [
            "FeatureGroup_RetGrLvl",
            "FeatureGroup_VolGrLvl",
            "FeatureGroup_RetGrRk",
            "FeatureGroup_VolGrRk",
            "FeatureGroup_AvgReturnPct",
            "FeatureGroup_WeightedIndexPct"
        ]
        
        for lag in self.lagList:
            featureNames.extend([
                f"FeatureGroup_RetGrLvl_lag_m{lag}",
                f"FeatureGroup_VolGrLvl_lag_m{lag}",
                f"FeatureGroup_RetGrRk_lag_m{lag}",
                f"FeatureGroup_VolGrRk_lag_m{lag}",
                f"FeatureGroup_AvgReturnPct_lag_m{lag}",
                f"FeatureGroup_WeightedIndexPct_lag_m{lag}"
            ])
        
        for lag in self.lagList:
            for m in self.monthHorizonList:
                featureNames.append(f"FeatureGroup_WeightedIndexMHPct_lag_m{lag}_MH_{m}")
        
        return featureNames