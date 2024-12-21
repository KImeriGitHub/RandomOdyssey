import numpy as np
import pandas as pd
import polars as pl
from typing import Dict
import bisect
import holidays

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.mathTools.TAIndicators import TAIndicators
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

from src.featureAlchemy.FeatureFourierCoeff import FeatureFourierCoeff

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
                 params: dict = None):
        
        # Update default parameters with any provided parameters
        self.params = self.DEFAULT_PARAMS
        if params is not None:
            self.params.update(params)

        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.monthsHorizon = self.params['monthsHorizon']
        self.timesteps = self.params['timesteps'] 
        
        featureFourier = FeatureFourierCoeff(asset)

        
        # NOTE: Implement someground feature like price and revenue without scaling
        # TODO: Ranking maybe
        # TODO: LAG