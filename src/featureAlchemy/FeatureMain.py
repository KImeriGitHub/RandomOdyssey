import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.mathTools.TAIndicators import TAIndicators
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

from src.featureAlchemy.FeatureFourierCoeff import FeatureFourierCoeff
from src.featureAlchemy.FeatureCategory import FeatureCategory
from src.featureAlchemy.FeatureFinancialData import FeatureFinancialData
from src.featureAlchemy.FeatureMathematical import FeatureMathematical
from src.featureAlchemy.FeatureSeasonal import FeatureSeasonal
from src.featureAlchemy.FeatureTA import FeatureTA

class FeatureMain():
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 15,
        'multFactor': 8,
        'monthsHorizon': 13,
        'timesteps': 5,
    }

    def __init__(self, 
                 asset: AssetDataPolars,
                 startDate: pd.Timestamp, 
                 endDate:pd.Timestamp, 
                 lagList: List[int],
                 timeLagList = [],
                 params: dict = None,
                 enableTimeSeries = True):
        
        self.asset = asset
        self.startDate = startDate
        self.endDate = endDate
        self.lagList = lagList
        self.timeLagList = timeLagList
        self.enableTimeSeries = enableTimeSeries
        
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.monthsHorizon = self.params['monthsHorizon']
        self.timesteps = self.params['timesteps']
        
        self.featCategory = FeatureCategory(asset)
        self.featMathematical = FeatureMathematical(asset, self.lagList)
        self.featFourierCoeff = FeatureFourierCoeff(asset, self.startDate, self.endDate, self.lagList, self.params)
        self.featFinancialData = FeatureFinancialData(asset, self.lagList)
        self.featSeasonal = FeatureSeasonal(asset, self.startDate, self.endDate, self.lagList, self.params)
        self.featTA = FeatureTA(asset, self.startDate, self.endDate, self.lagList)
        
        if self.enableTimeSeries:
            self.featCategory_timelag = self.featCategory
            self.featMathematical_timelag = FeatureMathematical(asset, self.timeLagList) 
            self.featFourierCoeff_timelag = FeatureFourierCoeff(asset, self.startDate, self.endDate, self.timeLagList, self.params) 
            self.featFinancialData_timelag = FeatureFinancialData(asset, self.timeLagList) 
            self.featSeasonal_timelag = FeatureSeasonal(asset, self.startDate, self.endDate, self.timeLagList, self.params) 
            self.featTA_timelag = FeatureTA(asset, self.startDate, self.endDate, self.timeLagList) 

    def getFeatureNames(self) -> list[str]:
        return (
            self.featCategory.getFeatureNames()+
            self.featMathematical.getFeatureNames()+
            self.featFourierCoeff.getFeatureNames()+
            self.featFinancialData.getFeatureNames()+
            self.featSeasonal.getFeatureNames()+
            self.featTA.getFeatureNames()
        )
        
    def getTimeFeatureNames(self) -> list[str]:
        if not self.enableTimeSeries:
            return []
        
        return (
            self.featCategory.getFeatureNames()+
            self.featMathematical.getFeatureNames()+
            self.featFourierCoeff.getFeatureNames()+
            self.featFinancialData.getFeatureNames()+
            self.featSeasonal.getFeatureNames()+
            self.featTA.getFeatureNames()
        )
    
    def apply(self, date: pd.Timestamp, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(date)
        
        niveau = 1.0
        # Get features for current date
        features = self.featCategory.apply(niveau)
        features = np.concatenate((features, self.featSeasonal.apply(date, niveau)))
        features = np.concatenate((features, self.featMathematical.apply(date, niveau, idx)))
        features = np.concatenate((features, self.featFourierCoeff.apply(date, niveau, idx)))
        features = np.concatenate((features, self.featFinancialData.apply(date, niveau, idx)))
        features = np.concatenate((features, self.featTA.apply(date, niveau, idx)))
        
        return features
        
    def apply_timeseries(self, date: pd.Timestamp, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DPl(self.asset.adjClosePrice).getNextLowerOrEqualIndex(date)
            
        if not self.enableTimeSeries:
            return np.array([])
        
        # Get features of timeseries: a matrix where each row is a feature vector of a time step
        featureMatrix = np.zeros((self.timesteps, len(self.getFeatureNames())))
        
        #reverse the order of the timeLagList
        for idx_t in range(idx-self.timesteps+1, idx+1):
            niveau = self.asset.adjClosePrice['AdjClose'].item(idx_t)
            timefeatures = self.featCategory.apply(niveau)
            timefeatures = np.concatenate((timefeatures, self.featSeasonal_timelag.apply(date, niveau)))
            timefeatures = np.concatenate((timefeatures, self.featMathematical_timelag.apply(date, niveau, idx_t)))
            timefeatures = np.concatenate((timefeatures, self.featFourierCoeff_timelag.apply(date, niveau, idx_t)))
            timefeatures = np.concatenate((timefeatures, self.featFinancialData_timelag.apply(date, niveau, idx_t)))
            timefeatures = np.concatenate((timefeatures, self.featTA_timelag.apply(date, niveau, idx_t)))
            
            featureMatrix[idx_t-idx+self.timesteps-1,:] = timefeatures
            
        return featureMatrix
            
            