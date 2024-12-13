import numpy as np
import pandas as pd
import polars as pl
from typing import Dict
import bisect
import holidays

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion

class FeatureFourierCoeff():
    def __init__(self, asset: AssetDataPolars, multFactor, fourierCutoff):
        self.multFactor = multFactor
        self.fourierCutoff = fourierCutoff
        self.asset = asset
        self.FourierPreMatrix = self.__preprocess_fourierConst(asset)

    def __preprocess_fourierConst(self):
        
        relDiffPerStep, res_cos, res_sin = SeriesExpansion.getFourierInterpCoeff(pastPrices, self.multFactor, self.fourierCutoff)
        
        resarr, rsme = SeriesExpansion.getFourierInterpFunct(res_cos, res_sin, pastPrices)
        
        features = []
        features.extend(res_cos[1:])
        features.extend(res_sin[1:])
        features.append(rsme)
        
        return features