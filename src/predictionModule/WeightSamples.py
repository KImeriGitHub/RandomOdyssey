import numpy as np
from src.mathTools.DistributionTools import DistributionTools

import logging
logger = logging.getLogger(__name__)

class WeightSamples:
    default_params= {
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
            feat_train: np.ndarray,
            y_train: np.ndarray,
            treenames: list[str],
            feat_test: np.ndarray,
            params: dict | None = None
        ):
        self.feat_train = feat_train
        self.y_train = y_train
        self.feat_test = feat_test
        self.treenames = treenames
        
        self.params = {**self.default_params, **(params or {})}

        self.__simple_tests()

    def __simple_tests(self) -> None:
        """
        Perform simple tests on the training datasets.
        """
        if self.feat_train.shape[1] != len(self.treenames):
            logger.error("Number of features in training data does not match the number of tree names.")

    def establish_ksDistance(self) -> np.ndarray:
        nSamples = self.feat_train.shape[0]
        min_n_samples = int(1e4)
        sparse_ratio = self.params['WeightSamples_sparsesamples_ratio']
        mask_sparsing = np.random.rand(nSamples) <= max(sparse_ratio, min_n_samples/nSamples)

        ksDist = DistributionTools.ksDistance(
            self.feat_train[mask_sparsing].astype(np.float64),
            self.feat_test.copy().astype(np.float64),
            weights=None,
            overwrite=True
        )
        
        logger.info(f"  Train-Test Distri Equality: Mean: {np.mean(ksDist)}, Quantile 0.9: {np.quantile(ksDist, 0.9)}")

    def establish_matching_featureindices(self, ksDist) -> np.ndarray:
        
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

    def establish_weights(self, n_bin: int = 15, min_bd: float = 0.1, wndw_ratio: float = 0.2) -> np.ndarray:
        """
        Establish weights for the training samples based on their importance.
        """
        tree_weights = DistributionTools.establishMatchingWeight(
            self.feat_train.astype(np.float64),
            self.feat_test.astype(np.float64),
            n_bin = n_bin,
            minbd = min_bd,
            wndw_ratio = wndw_ratio
        )
        tree_weights *= (self.feat_train.shape[0] / np.sum(tree_weights))

        logger.debug(f"  Zeros Weight Ratio: {np.sum(tree_weights < 1e-6) / len(tree_weights)}")
        logger.debug(f"  Negative Weight Ratio: {np.sum(tree_weights < -1e-5) / len(tree_weights)}")
        logger.debug(f"  Mean Weight: {np.mean(tree_weights)}")
        logger.debug(f"  Quantile 0.1 Weight: {np.quantile(tree_weights, 0.1)}")
        logger.debug(f"  Quantile 0.9 Weight: {np.quantile(tree_weights, 0.9)}")
        
        return tree_weights