import numpy as np
import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetDataPolars import AssetDataPolars

class FeatureCategory():
    operator = "alphavantage"
    
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
    
    def __init__(self, asset: AssetDataPolars):
        self.asset = asset
        
        self.cat = self.cat_alphavantage if self.operator == "alphavantage" else self.cat_yfinance
    
    def getFeatureNames(self) -> list[str]:
        features_names = ["Category_" + val for val in self.cat]
            
        return features_names
    
    def apply(self, scaleToNiveau: float):
        sector = self.asset.sector
        
        # Create a one-hot encoding where the category matches the sector
        features = np.array([1.0 if category == sector else 0.0 for category in self.cat])
        
        return features*scaleToNiveau