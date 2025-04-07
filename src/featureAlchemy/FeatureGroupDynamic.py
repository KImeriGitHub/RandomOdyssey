import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List

from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

class FeatureGroupDynamic():
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'timesteps': 10,
    }

    def __init__(self, 
            assetspl: Dict[str, AssetDataPolars], 
            startDate: pd.Timestamp, 
            endDate:pd.Timestamp, 
            lagList: List[int] = [],
            monthHorizonList: List[int] = [],
            params: dict = None):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        
                # --- Input Validation Asserts ---
        assert isinstance(assetspl, dict) and assetspl, "assetspl must be a non-empty dictionary"
        assert all(isinstance(ticker, str) for ticker in assetspl.keys()), "Asset keys must be strings (tickers)"
        assert all(isinstance(asset, AssetDataPolars) for asset in assetspl.values()), "Asset values must be AssetDataPolars instances"
        assert isinstance(startDate, pd.Timestamp), "startDate must be a pandas Timestamp"
        assert isinstance(endDate, pd.Timestamp), "endDate must be a pandas Timestamp"
        assert startDate <= endDate, "startDate must be less than or equal to endDate"
        assert isinstance(lagList, list) and all(isinstance(lag, int) and lag > 0 for lag in lagList), "lagList must be a list of positive integers"
        # assert isinstance(monthHorizonList, list) and all(isinstance(h, int) and h > 0 for h in monthHorizonList), "monthHorizonList must be a list of positive integers"
        
        self.assetspl = assetspl
        self.startDate = startDate
        self.endDate = endDate
        self.lagList = lagList
        self.monthHorizonList = monthHorizonList

        self.tickers = list(self.assetspl.keys())
        self.nAssets = len(self.tickers)
        
        self.timeseries_ivalList = [3,7]

        self.timesteps = self.params['timesteps']
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        
        #preprocess
        self.business_days = pd.date_range(start=startDate, end=endDate, freq="B")
        first_bd = self.business_days[0]
        last_bd = self.business_days[-1]
        self.nDates = len(self.business_days)
                
        idxAssets_start = {}
        idxAssets_end = {}
        for ticker, _ in self.assetspl.items():
            idxAssets_start[ticker] = DPl(self.assetspl[ticker].shareprice).getNextLowerOrEqualIndex(first_bd)
            idxAssets_end[ticker] = DPl(self.assetspl[ticker].shareprice).getNextLowerOrEqualIndex(last_bd)
            
        assert (
            np.all([(idxAssets_end[ticker]-idxAssets_start[ticker]+1) 
                    == self.nDates for ticker in idxAssets_start.keys()]), 
            "All assets must have the same number of dates.")
        
        self.avgVolume = np.zeros((self.nDates))
        self.minVolume = np.zeros((self.nDates))
        self.maxVolume = np.zeros((self.nDates))
        self.avgReturn = np.zeros((self.nDates))
        self.minReturn = np.zeros((self.nDates))
        self.maxReturn = np.zeros((self.nDates))
        self.avgWgtPrice = np.zeros((self.nDates))
        
        weightPerAsset = {}  #TODO: ideally using outstanding shares instead of volume
        for ticker in idxAssets_start.keys():
            weightPerAsset[ticker] = (
                self.assetspl[ticker].shareprice["Volume"]
                .rolling_mean(window_size = 2*self.idxLengthOneMonth)
                .to_numpy()
            )
        sumPerDate = np.zeros((self.nDates))
        for i, date in enumerate(self.business_days):
            sumPerDate[i] = np.sum([weightPerAsset[ticker][idxAssets_start[ticker] + i] for ticker in idxAssets_start.keys()])
        for ticker in idxAssets_start.keys():
            for i, date in enumerate(self.business_days):
                assetIdx = idxAssets_start[ticker] + i
                weightPerAsset[ticker][assetIdx] = weightPerAsset[ticker][assetIdx] / sumPerDate[i]
        
        #Group Dynamics
        for i, date in enumerate(self.business_days):
            self.minVolume[i] = np.min([self.assetspl[ticker].shareprice["Volume"].item(idxAssets_start[ticker] + i) for ticker in idxAssets_start.keys()])
            self.maxVolume[i] = np.max([self.assetspl[ticker].shareprice["Volume"].item(idxAssets_start[ticker] + i) for ticker in idxAssets_start.keys()])
            self.avgVolume[i] = np.mean([self.assetspl[ticker].shareprice["Volume"].item(idxAssets_start[ticker] + i) for ticker in idxAssets_start.keys()])
            self.minReturn[i] = np.min([self.assetspl[ticker].shareprice["Close"].pct_change().item(idxAssets_start[ticker] + i) for ticker in idxAssets_start.keys()])
            self.maxReturn[i] = np.max([self.assetspl[ticker].shareprice["Close"].pct_change().item(idxAssets_start[ticker] + i) for ticker in idxAssets_start.keys()])
            self.avgReturn[i] = np.mean([self.assetspl[ticker].shareprice["Close"].pct_change().item(idxAssets_start[ticker] + i) for ticker in idxAssets_start.keys()])
            self.avgWgtPrice[i] = np.mean([self.assetspl[ticker].shareprice["Close"].pct_change().item(idxAssets_start[ticker] + i) 
                * weightPerAsset[ticker][idxAssets_start[ticker] + i] for ticker in idxAssets_start.keys()])
        
        featureDict: Dict[str, List[np.array]] = {}
        features: List[np.array] = []
        featuresNames = []
        for ticker, asset in self.assetspl.items():
            closePct: np.array = asset.shareprice["Close"].pct_change().slice(idxAssets_start[ticker], self.nDates).to_numpy()
            curVol: np.array = asset.shareprice["Volume"].slice(idxAssets_start[ticker], self.nDates).to_numpy()
            features[0], featuresNames[0] = (closePct-self.minReturn) / (self.maxReturn-self.minReturn), "FeatureGroup_RetGrLevel"
            features[1], featuresNames[1] = (curVol-self.minVolume) / (self.maxVolume-self.minVolume), "FeatureGroup_VolGrLevel"
            features[2], featuresNames[2] = self.avgReturn, "FeatureGroup_AvgReturn"
            features[3], featuresNames[3] = self.avgWgtPrice, "FeatureGroup_AvgWgtPrice"
            
            featureDict[asset].extend(features)
        
        

        #TODO: Cointegration features, lagged features
        
        # asserts
        assert np.all([self.nDates == len(self.assetpl[asset].shareprice["Close"]) for asset in self.assetpl.values()]), "All assets must have the same number of dates."

        
    def getFeatureNames(self) -> list[str]:
        featureNames = [
            'MathFeature_TradedPrice_log',
            "MathFeature_Price_Diff",
            "MathFeature_Price_DiffDiff",
            "MathFeature_Price_logDiff",
            "MathFeature_Price_logDiffDiff",
            "MathFeature_Return",
            "MathFeature_Return_log",
            "MathFeature_PriceAdjustment",
        ]

        for m in self.monthHorizonList:
            featureNames.extend([
                f"MathFeature_Drawdown_MH{m}",
                f"MathFeature_Drawup_MH{m}",
            ])
        
        for lag in self.lagList:
            featureNames.extend([
                f"MathFeature_Price_Diff_lag_m{lag}",
                f"MathFeature_Price_DiffDiff_lag_m{lag}",
                f"MathFeature_Price_logDiff_lag_m{lag}",
                f"MathFeature_Price_logDiffDiff_lag_m{lag}",
                f"MathFeature_Return_lag_m{lag}",
                f"MathFeature_Return_log_lag_m{lag}",
                f"MathFeature_PriceAdjustment_lag_m{lag}",
            ])

            for m in self.monthHorizonList:
                featureNames.extend([
                    f"MathFeature_Drawdown_lag_m{lag}_MH{m}",
                    f"MathFeature_Drawup_lag_m{lag}_MH{m}",
                ])
        return featureNames
    
    def getTimeFeatureNames(self) -> list[str]:
        featureNames = ['MathFeature_TradedPrice']
        for i in range(len(self.timeseries_ivalList)):
            featureNames.append(f"MathFeature_TradedPrice_sp{i}")

        featureNames.extend([
            "MathFeature_Return",
            "MathFeature_PriceAdjustment"
        ])

        return featureNames
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(date)
        if idx-max(self.lagList, default=0) < 0 + 4:
            raise ValueError("Lag is too far back.")
        
        niveau = self.prices.item(idx)
        scalingfactor = scaleToNiveau/niveau
        
        mathFeatures = []  # Todo: make immutable
        mathFeatures.append(self.tradedPrice_log.item(idx))
        mathFeatures.extend([
            (self.prices_Diff.item(idx)) * scaleToNiveau, 
            (self.prices_DiffDiff.item(idx)) * scaleToNiveau, 
            (self.prices_logDiff.item(idx)) * scaleToNiveau, 
            (self.prices_logDiffDiff.item(idx)) * scaleToNiveau, 
            (self.pricesReturns.item(idx) * niveau) * scalingfactor, 
            (self.pricesReturns_log.item(idx) * niveau) * scalingfactor, 
            (self.priceAdjustments.item(idx) * niveau) * scalingfactor,
        ])

        for i, _ in enumerate(self.monthHorizonList):
            mathFeatures.extend([
                (self.drawdown[i].item(idx) * niveau) * scalingfactor,
                (self.drawup[i].item(idx) * niveau) * scalingfactor,
            ])
        
        for lag in self.lagList:
            idx_lag = idx - lag
            mathFeatures.extend([
                (self.prices_Diff.item(idx_lag)) * scaleToNiveau, 
                (self.prices_DiffDiff.item(idx_lag)) * scaleToNiveau, 
                (self.prices_logDiff.item(idx_lag)) * scaleToNiveau, 
                (self.prices_logDiffDiff.item(idx_lag)) * scaleToNiveau, 
                (self.pricesReturns.item(idx_lag) * niveau) * scalingfactor, 
                (self.pricesReturns_log.item(idx_lag) * niveau) * scalingfactor,
                (self.priceAdjustments.item(idx_lag) * niveau) * scalingfactor,
            ])

            for i, m in enumerate(self.monthHorizonList):
                mathFeatures.extend([
                    (self.drawdown[i].item(idx_lag) * niveau) * scalingfactor,
                    (self.drawup[i].item(idx_lag) * niveau) * scalingfactor,
                ])
        
        features = np.array(mathFeatures)
        
        return features.astype(np.float32)
    
    def apply_timeseries(self, date: pd.Timestamp, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(date)
        if idx-max(self.lagList, default=0) < 0 + 4:
            raise ValueError("Lag is too far back.")
        if idx - self.timesteps * np.max(self.timeseries_ivalList) < 0:
            raise ValueError("Not enough data for time series.")
        
        coreLen = len(self.getTimeFeatureNames())
        featuresMat = np.zeros((self.timesteps, coreLen))

        adjFactor = self.tradedPrice.item(idx) / self.prices.item(idx) # to avoid leakage
        niveau = self.tradedPrice.item(idx)
        for ts in range(0, self.timesteps):
            idx_ts = idx - (self.timesteps - 1) + ts

            featuresMat[ts, 0] = np.tanh(self.prices.item(idx_ts) * adjFactor / niveau - 1.0)/2.0 + 0.5
            for i, sp in enumerate(self.timeseries_ivalList):
                idx_ts_sp = idx - ((self.timesteps - 1) - ts) * sp
                featuresMat[ts, i+1] = np.tanh(self.prices.item(idx_ts_sp) * adjFactor / niveau - 1.0)/2.0 + 0.5

            featuresMat[ts, coreLen-2] = np.tanh(self.pricesReturns.item(idx_ts)-1.0)/2.0+0.5
            featuresMat[ts, coreLen-1] = np.tanh(self.priceAdjustments.item(idx_ts))/2.0+0.5

        return featuresMat.astype(np.float32)