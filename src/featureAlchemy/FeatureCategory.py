import numpy as np
import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetDataPolars import AssetDataPolars
from src.common.YamlTickerInOut import YamlTickerInOut

class FeatureCategory():
    operator = "alphavantage"
    
    DEFAULT_PARAMS = {
        'timesteps': 10,
    }
    
    # Class-level default parameters
    cat_alphavantage = [
        'other', 
        'industrials',
        'healthcare', 
        'technology', 
        'financial-services', 
        'real-estate', 
        'energy', 
        'consumer-cyclical', 
    ]
    
    cat_yfinance =[
        'other', 'industrials', 'healthcare', 'technology', 'utilities', 
        'financial-services', 'basic-materials', 'real-estate', 
        'consumer-defensive', 'energy', 'communication-services', 
        'consumer-cyclical'
    ]
    
    def __init__(self, asset: AssetDataPolars, params: dict = None):
        self.asset = asset
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.timesteps = self.params['timesteps']
        
        self.cat = self.cat_alphavantage if self.operator == "alphavantage" else self.cat_yfinance
        self.snp500tickers: list[str] = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]
        self.nas100tickers: list[str] = YamlTickerInOut("src/tickerSelection").loadFromFile("nas100.yaml")["nas100tickers"]
    
    def getFeatureNames(self) -> list[str]:
        features_names = ["Category_" + val for val in self.cat]
        features_names += ["Category_inSnP500", "Category_inNas100"]
            
        return features_names
    
    def getTimeFeatureNames(self) -> list[str]:
        return []
    
    def apply(self, scaleToNiveau: float):
        sector = self.asset.sector
        
        # Create a one-hot encoding where the category matches the sector
        features = np.zeros(len(self.cat)+2, dtype=np.float32)
        features[:len(self.cat)] = np.array([1.0 if category == sector else 0.0 for category in self.cat])
        features[len(self.cat)] = self.asset.ticker in self.snp500tickers
        features[len(self.cat)+1] = self.asset.ticker in self.nas100tickers
        
        return features*scaleToNiveau
    
    def apply_timeseries(self, date: pd.Timestamp, idx: int = None) -> np.ndarray:
        return np.empty((self.timesteps, 0), dtype=np.float32)