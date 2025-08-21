import numpy as np
import polars as pl
import datetime
from typing import Optional
import torch
from tqdm import tqdm
import random
from src.mathTools.DistributionTools import DistributionTools

import logging
logger = logging.getLogger(__name__)

class WeightSamples:
    default_params= {
        "WeightSamples_minWeight": 0.4,
        "WeightSamples_truncation": 2,
        "WeightSamples_sparsesamples_ratio": 0.1,
        "WeightSamples_Pricediff": True,
        "WeightSamples_FinData_quar": False,
        "WeightSamples_FinData_metrics": False,
        "WeightSamples_Fourier_RSME": False,
        "WeightSamples_Fourier_Sign": False,
        "WeightSamples_TA_trend": False,
        "WeightSamples_FeatureGroup_VolGrLvl": False,
        "WeightSamples_LSTM_Prediction": False,
    }
    
    def __init__(self,
            Xtree_train: np.ndarray,
            ytree_train: np.ndarray,
            treenames: list[str],
            Xtree_test: np.ndarray,
            params: dict | None = None
        ):
        self.Xtree_train = Xtree_train
        self.ytree_train = ytree_train
        self.Xtree_test = Xtree_test
        self.treenames = treenames
        
        self.params = {**self.default_params, **(params or {})}
        
        self.ksDist = self.__establish_ksDistance()
        self.top_idces = self.__establish_matching_featureindices(self.ksDist)

        self.__simple_tests()

    def __simple_tests(self) -> None:
        """
        Perform simple tests on the training datasets.
        """
        if self.Xtree_train.shape[1] != len(self.treenames):
            logger.error("Number of features in training data does not match the number of tree names.")

    def __establish_ksDistance(self) -> np.ndarray:
        nSamples = self.Xtree_train.shape[0]
        min_n_samples = int(1e4)
        sparse_ratio = self.params['WeightSamples_sparsesamples_ratio']
        mask_sparsing = np.random.rand(nSamples) <= max(sparse_ratio, min_n_samples/nSamples)

        ksDist = DistributionTools.ksDistance(
            self.Xtree_train[mask_sparsing].astype(np.float64),
            self.Xtree_test.copy().astype(np.float64),
            weights=None,
            overwrite=True
        )
        
        logger.info(f"  Train-Test Distri Equality: Mean: {np.mean(ksDist)}, Quantile 0.9: {np.quantile(ksDist, 0.9)}")

    def __establish_matching_featureindices(self, ksDist) -> np.ndarray:
        
        nFeat = len(self.treenames)
        mask_colToMatch = np.zeros(nFeat, dtype=bool)
        
        if self.params["WeightSamples_Pricediff"]:
            mask_colToMatch |= np.char.find(self.treenames, "MathFeature_Price_Diff") >= 0

        if self.params["WeightSamples_FinData_quar"]:
            mask_colToMatch |= np.char.find(self.treenames, "FinData_quar") >= 0

        if self.params["WeightSamples_FinData_metrics"]:
            mask_colToMatch |= np.char.find(self.treenames, "FinData_metrics") >= 0

        if self.params["WeightSamples_Fourier_RSME"]:
            mask_colToMatch |= np.char.find(self.treenames, "Fourier_Price_RSME") >= 0

        if self.params["WeightSamples_Fourier_Sign"]:
            mask_colToMatch |= np.char.find(self.treenames, "Fourier_Price_Sign") >= 0
        
        if self.params["WeightSamples_TA_trend"]:
            mask_colToMatch |= np.char.find(self.treenames, "FeatureTA_trend") >= 0
        
        if self.params["WeightSamples_FeatureGroup_VolGrLvl"]:
            mask_colToMatch |= np.char.find(self.treenames, "FeatureGroup_VolGrLvl") >= 0
        
        if self.params["WeightSamples_LSTM_Prediction"]:
            mask_colToMatch |= np.char.find(self.treenames, "LSTM_Prediction") >= 0
        
        if all(~mask_colToMatch):
            mask_colToMatch = np.char.find(self.treenames, "MathFeature_Price_Diff") >= 0
        
        idces = np.arange(mask_colToMatch.shape[0])[mask_colToMatch]
        idces = idces[np.argsort(ksDist[mask_colToMatch])]
        top_idces = idces[-self.params['WeightSamples_truncation']:]
            
        return top_idces

    def establish_weights(self) -> np.ndarray:
        """
        Establish weights for the training samples based on their importance.
        """
        tree_weights = DistributionTools.establishMatchingWeight(
            self.Xtree_train[:, self.top_idces].astype(np.float64),
            self.Xtree_test[:, self.top_idces].astype(np.float64),
            n_bin = 15,
            minbd = self.params['WeightSamples_minWeight']
        )
        tree_weights *= (self.Xtree_train.shape[0] / np.sum(tree_weights))

        logger.debug(f"  Zeros Weight Ratio: {np.sum(tree_weights < 1e-6) / len(tree_weights)}")
        logger.debug(f"  Negative Weight Ratio: {np.sum(tree_weights < -1e-5) / len(tree_weights)}")
        logger.debug(f"  Mean Weight: {np.mean(tree_weights)}")
        logger.debug(f"  Quantile 0.1 Weight: {np.quantile(tree_weights, 0.1)}")
        logger.debug(f"  Quantile 0.9 Weight: {np.quantile(tree_weights, 0.9)}")
        
        return tree_weights