import numpy as np
import pandas as pd
import polars as pl
import datetime
from typing import Dict, List

from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperations as DOps

class FeatureGroupDynamic():
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
            params: dict = None):
        
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
        
        exampleAsset = self.assetspl[self.tickers[0]]
        
        #preprocess
        self.business_days= pd.bdate_range(start=self.startRecord, end=self.endDate).date.tolist()  
        self.startBDate = self.business_days[0]
        self.endBDate = self.business_days[-1]
        self.nDates = len(self.business_days)
        
        self.idxAssets: Dict[str, List[int]] = {ticker: DOps(self.assetspl[ticker].shareprice).getNextLowerOrEqualIndices(self.business_days) for ticker in self.tickers}
            
        self.__preprocess()
        
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
        
    def apply(self, date: datetime.date, idx_dict: Dict[str, int] = None) -> Dict[str, np.ndarray]:
        if idx_dict is None:
            idx_dict: Dict[str, int] = {}
            for ticker, asset in self.assetspl.items():
                idx_dict[ticker] = DOps(asset.shareprice).getNextLowerOrEqualIndex(date)
                
        self.featureDict: Dict[str, np.array] = {}
        for ticker, asset in self.assetspl.items():
            features: List[float] = []
            idx_ticker_adj = idx_dict[ticker] - self.idxAssets[ticker][0]
            features.append(self.featureDict_RetGrLvl[ticker][idx_ticker_adj]) #"FeatureGroup_RetGrLvl"
            features.append(self.featureDict_VolGrLvl[ticker][idx_ticker_adj]) #"FeatureGroup_VolGrLvl"
            features.append(self.featureDict_RetGrRk[ticker][idx_ticker_adj]) #"FeatureGroup_RetGrRk"
            features.append(self.featureDict_VolGrRk[ticker][idx_ticker_adj]) #"FeatureGroup_VolGrRk"
            
            # Features Fixed over assets
            features.append(self.avgReturnPct[idx_ticker_adj]) #"FeatureGroup_AvgReturnPct"
            features.append(self.weightedIndex[idx_ticker_adj]/self.weightedIndex[max(idx_ticker_adj-1, 0)]-1) #"FeatureGroup_WeightedIndexPct"
            
            for lag in self.lagList:
                idx_ticker_adj_lag = idx_dict[ticker] - lag - self.idxAssets[ticker][0]
                features.append(self.featureDict_RetGrLvl[ticker][idx_ticker_adj_lag]) #"FeatureGroup_RetGrLvl_lag_m{lag}"
                features.append(self.featureDict_VolGrLvl[ticker][idx_ticker_adj_lag]) #"FeatureGroup_VolGrLvl_lag_m{lag}"
                features.append(self.featureDict_RetGrRk[ticker][idx_ticker_adj_lag]) #"FeatureGroup_RetGrRk_lag_m{lag}"
                features.append(self.featureDict_VolGrRk[ticker][idx_ticker_adj_lag]) #"FeatureGroup_VolGrRk_lag_m{lag}"
                
                features.append(self.avgReturnPct[idx_ticker_adj_lag]) #"FeatureGroup_AvgReturnPct_lag_m{lag}"
                features.append(self.weightedIndex[idx_ticker_adj_lag]/self.weightedIndex[max(idx_ticker_adj_lag-1, 0)]-1.0) #"FeatureGroup_WeightedIndexPct_lag_m{lag}"
                
            for lag in self.lagList:
                for m in self.monthHorizonList:
                    idx_ticker_adj_lag    = idx_dict[ticker] - self.idxAssets[ticker][0] - lag
                    idx_ticker_adj_lag_MH = idx_dict[ticker] - self.idxAssets[ticker][0] - lag - m * self.idxLengthOneMonth
                    features.append(self.weightedIndex[idx_ticker_adj_lag]/self.weightedIndex[max(idx_ticker_adj_lag_MH-1, 0)]-1.0) #"FeatureGroup_WeightedIndexMHPct_lag_m{lag}_MH_{m}"
            
            features = np.array(features, dtype=np.float32)
            self.featureDict[ticker] = features
        
        #TODO: Cointegration features, autocorrelation
        
        return self.featureDict
        
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
    
    def apply_timeseries(self, date: pd.Timestamp, idx_dict: Dict[str, int] = None) -> Dict[str,np.ndarray]:
        if idx_dict is None:
            idx_dict: Dict[str, int] = {}
            for ticker, asset in self.assetspl.items():
                idx_dict[ticker] = DOps(asset.shareprice).getNextLowerOrEqualIndex(date)
                
        self.featureDict: Dict[str, np.array] = {}
        for ticker, asset in self.assetspl.items():
            features: List[np.array] = []
            
            idx_ticker_adj = idx_dict[ticker] - self.idxAssets[ticker][0]
            s = idx_ticker_adj-self.timesteps+1
            e = idx_ticker_adj+1
            features.append(np.clip(self.featureDict_RetGrLvl[ticker][s:e],0.0,1.0)) #"FeatureGroup_RetGrLvl"
            features.append(np.clip(self.featureDict_VolGrLvl[ticker][s:e],0.0,1.0)) #"FeatureGroup_VolGrLvl"
            features.append(np.clip(self.featureDict_RetGrRk[ticker][s:e],0.0,1.0)) #"FeatureGroup_RetGrRk"
            features.append(np.clip(self.featureDict_VolGrRk[ticker][s:e],0.0,1.0)) #"FeatureGroup_VolGrRk"
            
            features.append(np.tanh(self.avgReturnPct[s:e])/2.0+0.5) #"FeatureGroup_AvgReturnPct"
            features.append(np.tanh(self.weightedIndex[s:e]/self.weightedIndex[s-1:e-1]-1.0)/2.0+0.5) #"FeatureGroup_WeightedIndexPct"
            features.append(np.tanh(self.weightedIndex[s:e]/self.weightedIndex[s-1-self.idxLengthOneMonth:e-1-self.idxLengthOneMonth]-1.0)/2.0+0.5) #"FeatureGroup_WeightedIndexPct_MH_1"
            
            features = np.array(features, dtype=np.float32)
            self.featureDict[ticker] = np.transpose(features)
            
        return self.featureDict
    
    def getTimeFeatureNames(self) -> list[str]:
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