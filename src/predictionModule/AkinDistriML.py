import numpy as np
import pandas as pd
import polars as pl
import bisect
from typing import Dict, List
import lightgbm as lgb
import optuna
import shap
import warnings
import re
import logging

from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from scipy import stats, spatial
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.mathTools.TAIndicators import TAIndicators
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.predictionModule.IML import IML
from src.featureAlchemy.FeatureMain import FeatureMain
from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.mathTools.DistributionTools import DistributionTools

class AkinDistriML(IML):
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 5,
        'multFactor': 6,
        'idxAfterPrediction': 21,
        'monthsHorizon': 13,
        'timesteps': 5,
        'classificationInterval': [0.05], 
    }

    def __init__(
            self, assets: Dict[str, AssetDataPolars], 
            test_date: pd.Timestamp,
            params: dict = None,
            gatherTestResults: bool = False,
            logger: logging.Logger = None
        ):
        
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.logger = logger
        
        self.lagList = [1,2,3,5,7,10,15,30,60,115,200,290]
        self.featureColumnNames = []
        self.gatherTestResults = gatherTestResults
        self.test_date = test_date
        
        trainingInterval_days = 365*4+60
        testInterval_idx = 2
        self.print_OptunaBars = True
        
        exampleAsset: AssetDataPolars = self.__assets[next(iter(self.__assets))]
        testDateIdx = DPl(exampleAsset.shareprice).getNextLowerOrEqualIndex(self.test_date)
        
        if not pd.Timestamp(exampleAsset.shareprice["Date"].item(testDateIdx)) == self.test_date:
            self.logger.info("test_start_date is not a business day. Correcting to first business day in assets before test_start_date.")
            self.test_date = pd.Timestamp(exampleAsset.shareprice["Date"].item(testDateIdx))
        
        self.testDates = pd.date_range(pd.Timestamp(exampleAsset.shareprice["Date"].item(testDateIdx+1-testInterval_idx)), self.test_date, freq='B')
        
        testDateIdx_m = testDateIdx - self.params['idxAfterPrediction']
        train_end_date = pd.Timestamp(exampleAsset.shareprice["Date"].item(testDateIdx_m))
        train_start_date = train_end_date - pd.Timedelta(days=trainingInterval_days)
        self.trainDates = pd.date_range(train_start_date, train_end_date, freq='B')
        
        assert exampleAsset.shareprice["Date"].last() >= max(self.testDates)
        
        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.idxAfterPrediction = self.params['idxAfterPrediction']
        self.monthsHorizon = self.params['monthsHorizon']
        self.classificationInterval = self.params['classificationInterval']
        self.timesteps = self.params['timesteps'] 
        
        # Store parameters in metadata
        self.metadata['Akin_params'] = self.params
        
        # Initialize instance variables for predicting
        self.lgbModelsList = None
        self.best_values = None
        
    def getTargetClassification(self, futureReturn: np.array, sorted_array: list[float]) -> list[float]:
        """
        Args:
            futureReturn (np.array): array of returns. (Ought to be around 0)
            sorted_array (list[float]): sorted list of floats. (Ought to be centered at 0)

        Returns:
            list[float]: for every entry in futureReturn, checks what the index of the next upper value in percInt is.
        """
        
        indices = [bisect.bisect_right(sorted_array, value) for value in futureReturn]

        return indices
    
    def getFeatures(self, asset: AssetDataPolars, featureMain: FeatureMain, date: pd.Timestamp, aidx: int):
        return featureMain.apply(date, idx=aidx)
    
    def getTarget(self, asset: AssetDataPolars, featureMain: FeatureMain, date: pd.Timestamp, aidx: int):
        if (aidx + self.idxAfterPrediction) >= len(asset.adjClosePrice["AdjClose"]):
            raise ValueError("Asset does not have enough data to calculate target.")
        
        curAdjPrice: float = asset.adjClosePrice["AdjClose"].item(aidx)
        
        futurePrice:float
        if self.params['target_option'] == 'last':
            futurePrice = asset.adjClosePrice["AdjClose"].item(aidx + self.idxAfterPrediction)
        if self.params['target_option'] == 'mean':
            futurePrice = asset.adjClosePrice["AdjClose"].slice(aidx+1, self.idxAfterPrediction).to_numpy().mean()
        if self.params['target_option'] == 'max':
            futurePrice = asset.adjClosePrice["AdjClose"].slice(aidx+1, self.idxAfterPrediction).to_numpy().max()
            
        futurePriceScaled = futurePrice/curAdjPrice
        
        target = self.getTargetClassification([futurePriceScaled-1], self.classificationInterval)
        target_reg = futurePriceScaled-1
        return target[0], target_reg

    def prepareData(self):
        Xtrain = []
        ytrain = []
        Xtest = []
        ytest = []
        Xval = []
        yval = []
        XtrainPrice = []
        ytrainPrice = []
        XtestPrice = []
        ytestPrice = []
        XvalPrice = []
        yvalPrice = []
        
        metaTrain = []
        metaVal = []
        metaTest = []
        
        processedCounter=0

        if self.trainDates is None \
             or self.testDates is None:
            raise ValueError("Data collection time is not defined.")
        if not (self.trainDates).intersection(self.testDates).empty:
            raise ValueError("There are overlapping dates between Train-Validation Dates and Test Dates.")

        #Main Loop
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice is None or not 'AdjClose' in asset.adjClosePrice.columns:
                continue

            self.logger.info(f"Processing asset: {asset.ticker}. Processed {processedCounter} out of {len(self.__assets)}.")
            processedCounter += 1
            
            params = {
                'idxLengthOneMonth': self.idxLengthOneMonth,
                'fouriercutoff': self.fouriercutoff,
                'multFactor': self.multFactor,
                'monthsHorizon': self.monthsHorizon,
                'timesteps': self.timesteps,
            }
            
            featureMain = FeatureMain(
                asset, 
                min([self.trainDates.min(),self.testDates.min()]), 
                max([self.trainDates.max(),self.testDates.max()]),
                lagList = self.lagList, 
                params=params,
                enableTimeSeries = False
            )
            
            if self.featureColumnNames == []:
                self.featureColumnNames = featureMain.getFeatureNames()
            elif self.featureColumnNames != featureMain.getFeatureNames():
                raise ValueError("Feature column names are not consistent across assets.")

            # Prepare Train Data
            for date in self.trainDates:
                aidx = DPl(asset.shareprice).getNextLowerOrEqualIndex(date)
                if asset.shareprice["Date"].item(aidx) != date:
                    continue
                if (aidx - self.monthsHorizon * self.idxLengthOneMonth - 1)<0:
                    self.logger.info("Warning! Asset History does not span far enough for features.")
                    continue
                if len(asset.shareprice["Date"]) <= aidx + self.idxAfterPrediction:
                    self.logger.info(f"Asset {ticker} does not have enough data to calculate target on date {date}.")
                    continue

                features = self.getFeatures(asset, featureMain, date, aidx)
                target, target_reg = self.getTarget(asset, featureMain, date, aidx)
                
                metaTrain.append([ticker, date])

                Xtrain.append(features)
                ytrain.append(target)
                ytrainPrice.append(target_reg)

            #Prepare Test Data
            for date in self.testDates:
                aidx = DPl(asset.shareprice).getNextLowerOrEqualIndex(date)
                if asset.shareprice["Date"].item(aidx) != date:
                    continue
                if (aidx - self.monthsHorizon * self.idxLengthOneMonth - 1)<0:
                    self.logger.info("Warning! Asset History does not span far enough for features.")
                    continue
                
                features = self.getFeatures(asset, featureMain, date, aidx)
                Xtest.append(features)
                metaTest.append([ticker, date])
                
                if self.gatherTestResults:
                    if len(asset.shareprice["Date"]) <= aidx + self.idxAfterPrediction:
                        raise ValueError(f"Asset {ticker} does not have enough data to calculate target on date: {date}.")
                    target, target_reg = self.getTarget(asset, featureMain, date, aidx)
                    ytest.append(target)
                    ytestPrice.append(target_reg)
                else:
                    ytest.append(-1)
                    ytestPrice.append(-1)

        self.X_train = np.array(Xtrain)
        self.y_train = np.array(ytrain).astype(int)
        
        self.X_val = np.array(Xval)
        self.y_val = np.array(yval).astype(int)
        
        self.X_test = np.array(Xtest)
        self.y_test = np.array(ytest).astype(int)
        
        self.meta_train = np.array(metaTrain)
        self.meta_test = np.array(metaTest)
        self.metaColumnNames = ['Ticker', 'Date']
        
        self.X_train_timeseries = np.array(XtrainPrice)
        self.y_train_timeseries = np.array(ytrainPrice)
        
        self.X_test_timeseries = np.array(XtestPrice)
        self.y_test_timeseries = np.array(ytestPrice)
        
        self.X_val_timeseries = np.array(XvalPrice)
        self.y_val_timeseries = np.array(yvalPrice)

        self.dataIsPrepared = True
        
    def __run_OptunaOnFiltered(self, mask_X_Train, mask_y_Train, mask_X_val, mask_y_val, mask_X_test, enablePrint: bool = False):
        def objective(trial: optuna.Trial):
            lgbm_params = {
                'verbosity': -1,
                'n_jobs': -1,
                'is_unbalance': True,
                'objective': 'binary',
                'metric': 'binary',
                'early_stopping_rounds': 2000//8,
                'num_boost_round': 20000//8,
                #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 100),
                #'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.001, 0.03),
                #'feature_fraction': trial.suggest_categorical('feature_fraction', [0.005,0.01,0.05]),
                'num_leaves': 2048//8, #trial.suggest_int('num_leaves', 512, 2048*2),
                'max_depth': trial.suggest_categorical('max_depth', [12, 25]),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'random_state': 41,
            }
            # Initialize and train LGBM model
            LGBMModel = lgb.LGBMClassifier(**lgbm_params)
            LGBMModel.fit(
                mask_X_Train, mask_y_Train,
                eval_set=[(mask_X_val, mask_y_val)]
            )
            mask_y_val_pred = LGBMModel.predict(mask_X_val)
            cm:np.array = confusion_matrix(mask_y_val_pred, mask_y_val, labels=np.unique(mask_y_val))
            cmsum = np.sum(cm, axis=1)
            per_class_accuracy = np.divide(
                np.diagonal(cm),
                cmsum,
                out=np.zeros_like(cmsum, dtype=float),
                where=(cmsum != 0)
            )
            overall_acc = accuracy_score(mask_y_val, mask_y_val_pred)
            return overall_acc # * np.sum(mask_y_test_pred == 1)

        # 3. Create a study2 object and optimize the objective function.
        #optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials = self.params['optuna_trials'], show_progress_bar=self.print_OptunaBars)
        if enablePrint:
            self.logger.info(f"  Best Value:  {study.best_value}")
            for key, value in study.best_trial.params.items():
                self.logger.info(f"  {key}: {value}")

        best_params  = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'objective': 'binary',
            'metric': 'binary',
            'early_stopping_rounds': 2000//8,
            'num_boost_round': 20000//8,
            #'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
            #'feature_fraction_bynode': study.best_trial.params['feature_fraction_bynode'],
            #'feature_fraction': study.best_trial.params['feature_fraction'],
            'num_leaves': 2048//8, #study.best_trial.params['num_leaves'],
            'max_depth': study.best_trial.params['max_depth'],
            'learning_rate': study.best_trial.params['learning_rate'],
            'random_state': 41,
        }
        lgbmInstance = lgb.LGBMClassifier(**best_params)
        #lgbmInstance.fit(X_final, y_final, sample_weight=w_final)        
        lgbmInstance.fit(mask_X_Train, mask_y_Train,
                         eval_set=[(mask_X_val, mask_y_val)])       
        return lgbmInstance, study.best_value
    
    def __run_LGBM(self, mask_X_Train, mask_y_Train, mask_X_val, mask_y_val, sample_weights):
        intscaler = 4
        best_params  = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'objective': 'binary',
            'metric': 'binary',
            'early_stopping_rounds': 2000//intscaler,
            'num_boost_round': 40000//intscaler,
            'min_data_in_leaf': self.params['Akin_min_data_in_leaf'],
            #'feature_fraction_bynode': study.best_trial.params['feature_fraction_bynode'],
            #'feature_fraction': study.best_trial.params['feature_fraction'],
            'num_leaves': 2048//intscaler, #study.best_trial.params['num_leaves'],
            'max_depth': self.params['Akin_max_depth'],
            'learning_rate': self.params['Akin_learning_rate'],
            'ma'
            'random_state': 41,
        }
        lgbmInstance = lgb.LGBMClassifier(**best_params)
        #lgbmInstance.fit(X_final, y_final, sample_weight=w_final)        
        lgbmInstance.fit(mask_X_Train, mask_y_Train,
                         eval_set=[(mask_X_val, mask_y_val)],
                         sample_weight=sample_weights)       
        return lgbmInstance
    
    def __run_LGBM_reg(self, 
            mask_X_Train, 
            mask_y_Train_reg, 
            mask_X_val, 
            mask_y_val_reg, 
            sample_weights,
            num_leaves:int = 512, 
            num_boost_round:int = 5000):

        train_data = lgb.Dataset(mask_X_Train, label = mask_y_Train_reg, weight=sample_weights)
        test_data = lgb.Dataset(mask_X_val, label = mask_y_val_reg, reference=train_data)
        
        params  = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'objective': 'regression',
            #'alpha': 0.85,
            'metric': 'l2_root',  # NOTE: the string 'rsme' is not recognized, v 4.5.0
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'early_stopping_rounds': num_boost_round//10,
            'feature_fraction': self.params['Akin_feature_fraction'],
            'num_leaves': num_leaves, 
            'max_depth': self.params['Akin_max_depth'],
            'learning_rate': self.params['Akin_learning_rate'],
            'min_data_in_leaf': self.params['Akin_min_data_in_leaf'],
            'min_gain_to_split': self.params['Akin_min_gain_to_split'],
            'path_smooth': self.params['Akin_path_smooth'],
            'min_sum_hessian_in_leaf': self.params['Akin_min_sum_hessian_in_leaf'],
            'random_state': 41,
        }   
        def print_eval_after_100(env):
            if env.iteration % 100 == 0 or env.iteration == num_boost_round:
                results = [
                    f"{data_name}'s {eval_name}: {result}"
                    for data_name, eval_name, result, _ in env.evaluation_result_list
                ]
                self.logger.info(f"Iteration {env.iteration}: " + ", ".join(results))
        gbm = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=num_boost_round,
            callbacks=[print_eval_after_100]
        )   
        return gbm
    
    def __get_ksDis(self, mask_train, mask_test, mask_features, weight: np.array = None):
        if weight is None:
            self.logger.info(f"  KS Distance calculation with no weight.")
            weight = np.zeros(len(mask_train))
            weight[mask_train] = 1.0
        weight *= (mask_train.sum()/np.sum(weight))
        
        mask_sparsing = np.random.rand(mask_train.sum()) <= (3e4/np.sum(mask_train)) # for speedup
        
        ksDist = DistributionTools.ksDistance(
            self.X_train[mask_train][mask_sparsing][:, mask_features], 
            self.X_test[mask_test][:, mask_features], 
            weights=weight[mask_train][mask_sparsing],
            overwrite=True)
            
        self.logger.info(f"  Train-Test Distri Equality: Mean: {np.mean(ksDist)}, Quantile 0.9: {np.quantile(ksDist, 0.9)}")
        
        res = np.zeros(len(mask_features), dtype=float)
        res[mask_features] = ksDist
        
        return res
        
    def __mask_weightedFeatures(self):
        feature_names = np.array(self.featureColumnNames).astype(str)
        patterns = [
            'Seasonal', 
            'Category', 
            'daysToReport', 
            '_rank', 
            '_Rank', 
            '_RANK',
            ]
        mask = np.ones(self.X_train.shape[1], dtype=bool)
        for i in range(self.X_train.shape[1]):
            if any([pattern in feature_names[i] for pattern in patterns]):
                mask[i] = False
                continue
            
            min_train = np.quantile(self.X_train[:,i],0.05)
            max_train = np.quantile(self.X_train[:,i],0.95)
            min_test = np.quantile(self.X_test[:,i],0.01)
            max_test = np.quantile(self.X_test[:,i],0.99)
            
            if max_train - min_train < 1e-4:  # TODO: There is one such thing. To investigate! Perhaps the year.
                mask[i] = False
            
            if max_train - min_train > 3.5 * (max_test - min_test):
                mask[i] = False
                
            if max_train < max_test or min_train > min_test:
                mask[i] = False
            
        return mask
        
    def __establishMask_rmOutliers(self, test_norm: np.array, rm_ratio: float):
        clf = IsolationForest(contamination=rm_ratio, random_state=41)
        clf.fit(test_norm)
        predictions = clf.predict(test_norm)
        return  (predictions > 0)
        
    def __establishWeights(self,
            mask_train, 
            mask_test,
            mask_features,
            n_trunc,
            n_bin = 20
        ):

        # Get distribution distances for each feature
        ksdis = self.__get_ksDis(mask_train, mask_test, mask_features)
        
        #mask_colToAssimilate = mask_features & self.__mask_weightedFeatures()
        mask_colToAssimilate = np.ones(len(mask_features), dtype=bool)
        
        # Get feature which we want to make the same distribution between train and test
        mask_MathFeature = np.char.find(self.featureColumnNames, "MathFeature_Price_Diff") >= 0
        mask_FinData_quar = (np.char.find(self.featureColumnNames, "FinData_quar_") >= 0)
        #mask_FinData_ann = (np.char.find(self.featureColumnNames, "FinData_ann_") >= 0)
        #mask_FinData_metrics = (np.char.find(self.featureColumnNames, "FinData_metrics_") >= 0)
        #mask_Fourier = (np.char.find(self.featureColumnNames, "Fourier_Price_AbsCoeff") >= 0)
        #mask_MathFeature = (np.char.find(self.featureColumnNames, "MathFeature_Return") >= 0)
        list_masks = [mask_MathFeature] #[mask_FeatureTA, mask_FinData_quar] #, mask_FinData_ann] # mask_FinData_metrics, mask_Fourier, mask_MathFeature]
        n_masks = len(list_masks)
        
        # Non lag exclusion
        mask_lagToExclude = np.zeros(len(mask_features), dtype=bool)
        for d in [i for i in self.lagList if i > 15]: 
            pattern = re.compile(rf"_m{d}(?!\d)")
            matches = np.array([bool(pattern.search(name)) for name in self.featureColumnNames])
            mask_lagToExclude |= matches
        mask_colToAssimilate = mask_colToAssimilate & (~mask_lagToExclude)
        
        # Get the top n_trunc features to assimilate for each mask
        splits: list = np.array_split(np.arange(n_trunc), n_masks)
        indices_mask = []
        for i in range(len(splits)):
            ksdis_i = ksdis.copy()
            ksdis_i *= 1.0/ksdis_i.max()
            mask_toSort = list_masks[i] & mask_colToAssimilate
            ksdis_i[~mask_toSort] = 0.0
            if ksdis_i.max() < 1e-1:
                ksdis_i[mask_toSort] = 1.0
            ksdis_argsort = np.argsort(ksdis_i)
            indices_mask.extend(ksdis_argsort[-len(splits[i]):])
        
        for i in range(len(indices_mask)):
            self.logger.info(f"  Feature {i}: {self.featureColumnNames[indices_mask[i]]}")
        
        sample_weights_loop = DistributionTools.establishMatchingWeight(
            self.X_train[mask_train][:, indices_mask],
            self.X_test[mask_test][:, indices_mask],
            n_bin = n_bin,
            minweight = 0.3
        )
        sample_weights = np.zeros(len(mask_train))
        sample_weights[mask_train] = sample_weights_loop
        sample_weights *= (np.sum(mask_train) / np.sum(sample_weights))
        
        self.__get_ksDis(mask_train, mask_test, mask_features, sample_weights)
        
        self.logger.info(f"  Zeros Weight Ratio: {np.sum(sample_weights[mask_train] < 1e-6) / len(sample_weights[mask_train])}")
        self.logger.info(f"  Negative Weight Ratio: {np.sum(sample_weights < -1e-5) / len(sample_weights)}") #For debugging
        return sample_weights
        
    def __establishAkinMask_Features(self, p_features: float, lgbmInstance: lgb.Booster, mask_train, mask_features):
        if p_features >= 1 - 1e-12:
            return mask_features
        
        # SHAP FEATURE IMPORTANCE
        #warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray")
        #explainer = shap.TreeExplainer(lgbmInstance)
        #shap_values = explainer.shap_values(self.X_train[mask_train][:, mask_features])
        #feature_importances = np.abs(shap_values).mean(axis=0)
        #shap_quant = np.quantile(feature_importances, 1-p_features)
        #
        #mask = feature_importances > shap_quant
        
        # BUILD-IN LGBM FEATURE IMPORTANCE
        feature_importances = lgbmInstance.feature_importance()
        n_keep = int(np.ceil(p_features * len(feature_importances)))
        sorted_idx = np.argsort(feature_importances)[::-1]  # descending order
        mask = np.zeros(len(feature_importances), dtype=bool)
        mask[sorted_idx[:n_keep]] = True
        feature_importances[~mask] = 0

        return mask, feature_importances
    
    def __establishAkinMask_Test(self, p_test, lgbmInstance: lgb.Booster, mask_test, mask_features):
        if p_test >= 1 - 1e-12:
            return np.ones(mask_test.sum(), dtype=bool)
        
        y_pred = lgbmInstance.predict(self.X_test[mask_test][:, mask_features], num_iteration=lgbmInstance.best_iteration)
        
        # Quantile regression to subset mask
        mask = y_pred >= np.quantile(y_pred, 1-p_test)
        return mask
    
    def __establishLGBMInstance(self, 
            mask_train, 
            mask_test, 
            mask_features, 
            feature_max:int, 
            sample_weights, 
            num_leaves:int = 256, 
            num_boost_round:int = 2500
        ):
        # yte_reg to be used with caution, because of leakage
        
        # Random subsample for validation set
        mask_val = (np.random.rand(len(mask_train)) < 0.05)
        if np.sum(mask_val) < 2:  
            ValueError("Not enough data points to establish validation set")
        mask_val = mask_val & mask_train
        
        train_data = lgb.Dataset(
            self.X_train[~mask_val][:, mask_features], 
            label = self.y_train_timeseries[~mask_val], 
            weight=sample_weights[~mask_val]
        )
        val_data = lgb.Dataset(
            self.X_train[mask_val][:, mask_features], 
            label = self.y_train_timeseries[mask_val], 
            reference=train_data
        )

        params = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'objective': 'regression',
            #'alpha': 0.85,
            'metric': 'l2_root',
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'early_stopping_rounds': num_boost_round//10,
            'feature_fraction': self.params['Akin_feature_fraction']*min([feature_max / mask_features.sum(), 1.0]),
            'num_leaves': num_leaves,
            'max_depth': self.params['Akin_max_depth'],
            'learning_rate': self.params['Akin_learning_rate'],
            'min_data_in_leaf': self.params['Akin_min_data_in_leaf'],
            'min_gain_to_split': self.params['Akin_min_gain_to_split'],
            'path_smooth': self.params['Akin_path_smooth'],
            'min_sum_hessian_in_leaf': self.params['Akin_min_sum_hessian_in_leaf'],
            'random_state': 41,
        }
        
        def print_eval_after_100(env):
            if env.iteration % 100 == 0 or env.iteration == num_boost_round:
                results = [
                    f"{data_name}'s {eval_name}: {result}"
                    for data_name, eval_name, result, _ in env.evaluation_result_list
                ]
                self.logger.info(f"Iteration {env.iteration}: " + ", ".join(results))
        
        gbm = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=[print_eval_after_100]
        )
        
        if self.gatherTestResults:
            self.logger.info(f"  Mask Testing Best Iteration: {gbm.best_iteration}")
            y_pred = gbm.predict(self.X_test[mask_test][:, mask_features], num_iteration=gbm.best_iteration)
            self.logger.info(f"  Accuracy at Test Mask: {np.abs(self.y_test_timeseries[mask_test]-y_pred).mean()}") 
        
        return gbm
    
    def applyFilter(self):
        # 0, 1, 2, 3, 5, 7, 10, 15, 30, 60, 115, 200, 290
        pHigh = [self.featureColumnNames.index("FeatureTA_High")]
        for lag in [i for i in self.lagList if i<70]:
            pHigh.append(self.featureColumnNames.index(f"FeatureTA_High_lag_m{lag}"))
        pHigh_lag290 = self.featureColumnNames.index("FeatureTA_High_lag_m290")
        
        pDiff = [self.featureColumnNames.index("MathFeature_Price_Diff")]
        for lag in [i for i in self.lagList if i<20]:  
            pDiff.append(self.featureColumnNames.index(f"MathFeature_Price_Diff_lag_m{lag}"))
            
        pDiffDiff = [self.featureColumnNames.index("MathFeature_Price_DiffDiff")]
        for lag in [i for i in self.lagList if i<6]:
            pDiffDiff.append(self.featureColumnNames.index(f"MathFeature_Price_DiffDiff_lag_m{lag}")) 
            
        pGrossProfit = []
        for lag in [i for i in self.lagList if i<7]:
            pGrossProfit.append(self.featureColumnNames.index(f"FinData_quar_grossProfit_lagquot_qm{lag}"))
            
        pebit = []
        for lag in [i for i in self.lagList if i<7]:
            pebit.append(self.featureColumnNames.index(f"FinData_quar_ebit_lagquot_qm{lag}"))    
            
        pFourierSignCoeff = self.featureColumnNames.index(f"Fourier_Price_SignCoeff_1")
        pFourierAbsCoeff = self.featureColumnNames.index(f"Fourier_Price_AbsCoeff_1")
            
        relPriceDiff_train: np.array = self.X_train[:, pDiff]
        relPriceDiff_test: np.array = self.X_test[:, pDiff]
        
        highPrice_train = self.X_train[:, pHigh]
        highPrice_test = self.X_test[:, pHigh]
        
        relPriceDiffDiff_train: np.array = self.X_train[:, pDiffDiff]
        relPriceDiffDiff_test: np.array = self.X_test[:, pDiffDiff]
            
        mask_train = np.ones(self.X_train.shape[0], dtype=bool)
        mask_test = np.ones(self.X_test.shape[0], dtype=bool)
        
        # Rising Action
        #mask_train &= np.all(relPriceDiff_train[:,0:3]>0.01, axis=1)
        #mask_test &= np.all(relPriceDiff_test[:,0:3]>0.01, axis=1)
        #
        #mask_train &= np.all(relPriceDiffDiff_train[:,0:2]>0.001, axis=1)
        #mask_test &= np.all(relPriceDiffDiff_test[:,0:2]>0.001, axis=1)
        #
        #q_minDiff_train = np.quantile(np.min(relPriceDiff_train, axis=1), 0.6)
        #q_minDiff_test = np.quantile(np.min(relPriceDiff_test[mask_test], axis=1), 0.95)
        #mask_train &= np.min(relPriceDiff_train, axis=1) > q_minDiff_test
        #mask_test &= np.min(relPriceDiff_test, axis=1) > q_minDiff_test
        #
        #q_minDiffDiff_train = np.quantile(np.min(relPriceDiffDiff_train, axis=1), 0.6)
        #q_minDiffDiff_test = np.quantile(np.min(relPriceDiffDiff_test, axis=1), 0.95)
        #mask_train &= np.min(relPriceDiffDiff_train, axis=1) > q_minDiffDiff_test
        #mask_test &= np.min(relPriceDiffDiff_test, axis=1) > q_minDiffDiff_test
        
        # Up Down
        #mask_train &= np.all(relPriceDiff_train[:,4]>0.01, axis=1)
        
        # High drop
        #mask_train &= highPrice_train[:,-1] > 1.11
        #mask_test &= highPrice_test[:,-1] > 1.11
        #
        #mask_train &= ((highPrice_train[:,6] > 0.98) 
        #               & (highPrice_train[:,6] < 1.02) 
        #               & (highPrice_train[:,4] < 0.95))
        #mask_test &= ((highPrice_test[:,6] > 0.98) 
        #              & (highPrice_test[:,6] < 1.02) 
        #              & (highPrice_test[:,4] < 0.95))
        
        # Drop with good earnings
        mask_train &= np.all(self.X_train[:,pebit] > 1.01, axis=1)
        mask_test &= np.all(self.X_test[:,pebit] > 1.01, axis=1)
        
        mask_train &= (self.X_train[:, pHigh_lag290] - highPrice_train[:,0]) > 0
        mask_test &= (self.X_test[:, pHigh_lag290] - highPrice_test[:,0]) > 0 
        
        #Fourier Coeffs
        #mask_train &= (self.X_train[:, pFourierSignCoeff] > 0.5)
        #mask_test &= (self.X_test[:, pFourierSignCoeff] > 0.5)
        #q_AbsCoeff = np.quantile(self.X_test[mask_test][:, pFourierAbsCoeff], 0.90)
        #mask_train &= (self.X_train[:, pFourierAbsCoeff] > q_AbsCoeff)
        #mask_test &= (self.X_test[:, pFourierAbsCoeff] > q_AbsCoeff)
        
        
        del relPriceDiff_train, relPriceDiff_test, relPriceDiffDiff_train, relPriceDiffDiff_test
        
        if mask_test.sum() < 2:
            mask_test = np.ones(self.X_test.shape[0], dtype=bool)
        
        return mask_train, mask_test
        
    def establishMasks(self, 
            q_test: float, 
            feature_max:int, 
            iterSteps: int,
            num_leaves:int = 256,
            num_boost_round:int = 5000,
            weight_truncation: int = 5,
            n_bin = 15,
            mask_train = None,
            mask_test = None
        ):
        if mask_train is None:
            mask_train = np.ones(self.X_train.shape[0], dtype=bool)
        if mask_test is None:
            mask_test = np.ones(self.X_test.shape[0], dtype=bool)
        mask_features = np.ones(self.X_train.shape[1], dtype=bool)
        sample_weights = np.ones(self.X_train.shape[0], dtype=float)
        feature_importances = np.ones(self.X_train.shape[1], dtype=float)
        
        p_test = q_test ** (1 / iterSteps)
        q_features = feature_max / self.X_train.shape[1]
        p_features = q_features ** (1 / iterSteps)
        
        for i in range(iterSteps):
            self.logger.info(f"Establish Mask: Step {i+1}/{iterSteps}.")
            startTime_loop = datetime.now()

            # Establish Weights
            sample_weights = self.__establishWeights(
                mask_train = mask_train, 
                mask_test = mask_test,
                mask_features = mask_features,
                n_trunc = weight_truncation,
                n_bin = n_bin
            )
            
            # Estblish LGBM Instance
            lgbmInstance = self.__establishLGBMInstance(
                mask_train = mask_train, 
                mask_test = mask_test,
                mask_features = mask_features,
                feature_max = feature_max,
                sample_weights = sample_weights,
                num_leaves = num_leaves, 
                num_boost_round = num_boost_round
            )
            
            # Establish Test Mask: quantil regression to subset mask
            mask_test_loop = self.__establishAkinMask_Test(
                p_test,
                lgbmInstance = lgbmInstance,
                mask_test = mask_test,
                mask_features = mask_features,
            )
            mask_test[mask_test] = mask_test_loop
            
            # Establish Feature Mask
            mask_features_loop, feature_importances_loop = self.__establishAkinMask_Features(
                p_features, 
                lgbmInstance = lgbmInstance,
                mask_train = mask_train,
                mask_features = mask_features,
            )
            feature_importances[mask_features] = feature_importances_loop
            mask_features[mask_features] = mask_features_loop

            endTime_loop = datetime.now()
            self.logger.info(f"  Time elapsed: {endTime_loop - startTime_loop}.")
            self.logger.info("  Masked Training Label Distribution:")
            ModelAnalyzer().print_label_distribution(self.y_train[mask_train], logger=self.logger)
            if self.gatherTestResults:
                self.logger.info("  Masked Test Label Distribution:")
                ModelAnalyzer().print_label_distribution(self.y_test[mask_test], logger=self.logger)
                
        # End of loop Establish Weights
        self.logger.info("End of Loop computation.")
        sample_weights = self.__establishWeights(
            mask_train = mask_train, 
            mask_test = mask_test,
            mask_features = mask_features,
            n_trunc = weight_truncation,
            n_bin = n_bin
        )
        
        return mask_train, mask_test, mask_features, sample_weights
    
    def analyze(self):
        if not self.gatherTestResults:
            raise ValueError("evaluateTestResults is set to False. Cannot analyze per filter.")
        if not self.dataIsPrepared:
            raise ValueError("Data is not prepared. Please run prepareData() first.")
        
        nTrain = self.X_train.shape[0]
        nTest = self.X_test.shape[0]
        
        test_quantil = self.params['Akin_test_quantile']
        feature_max = self.params['Akin_feature_max']
        itersteps = self.params['Akin_itersteps']
        num_leaves = self.params['Akin_pre_num_leaves']
        num_boost_round = self.params['Akin_pre_num_boost_round']
        weight_truncation = self.params['Akin_pre_weight_truncation']
        
        self.logger.info(f"num_leaves: {num_leaves}")
        self.logger.info(f"num_boost_round: {num_boost_round}")
        self.logger.info(f"weight_truncation: {weight_truncation}")
        self.logger.info(f"test_quantil: {test_quantil}")
        self.logger.info(f"feature_max: {feature_max}")
        self.logger.info(f"iterSteps: {itersteps}")
        
        # Start establishing masks
        startTime = datetime.now()
        mask_train, mask_test = self.applyFilter()
        
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        mask_train, mask_test, mask_features, sample_weights = self.establishMasks(
            q_test=test_quantil, 
            feature_max=feature_max, 
            iterSteps=itersteps,
            num_leaves = num_leaves,
            num_boost_round = num_boost_round,
            weight_truncation = weight_truncation,
            mask_train = mask_train,
            mask_test = mask_test
        )        
        endTime = datetime.now()
        self.logger.info(f"Masking completed in {endTime - startTime}.")
        
        self.logger.info(f"Training samples selected: {mask_train.sum()} out of {nTrain}")
        self.logger.info(f"Test samples selected: {mask_test.sum()} out of {nTest}")
        self.logger.info(f"Features selected: {mask_features.sum()} out of {self.X_test.shape[1]}")
        
        #Establish validation set
        # Random subsample for validation set
        mask_val = np.random.rand(np.sum(mask_train)) < 0.05
        if np.sum(mask_val) < 2:  
            ValueError("Not enough data points to establish validation set")
            
        self.X_val = self.X_train[mask_train][mask_val]
        self.y_val = self.y_train[mask_train][mask_val]
        self.y_val_reg = self.y_train_timeseries[mask_train][mask_val]
        
        masked_X_train = self.X_train[mask_train][~mask_val][:, mask_features]
        masked_y_train = self.y_train[mask_train][~mask_val]
        masked_y_train_reg = self.y_train_timeseries[mask_train][~mask_val]
        masked_X_val = self.X_val[:, mask_features]
        masked_y_val = self.y_val
        masked_y_val_reg = self.y_val_reg
        masked_X_test = self.X_test[mask_test][:, mask_features]
        masked_y_test = self.y_test[mask_test]
        masked_y_test_reg = self.y_test_timeseries[mask_test]
        
        masked_sample_weights_train = sample_weights[mask_train][~mask_val]
            
        self.logger.info(f"Number of features: {len(self.featureColumnNames)}")
        self.logger.info("Overall Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_train, logger = self.logger)
        self.logger.info("Overall Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_val, logger = self.logger)
        self.logger.info("Overall Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_test, logger = self.logger)
            
        if not len(np.unique(masked_y_train)) < 2:
            self.logger.info("Masked Training Label Distribution:")
            ModelAnalyzer().print_label_distribution(masked_y_train, logger = self.logger)
        if not len(np.unique(masked_y_val)) < 2:
            self.logger.info("Masked Validation Label Distribution:")
            ModelAnalyzer().print_label_distribution(masked_y_val, logger = self.logger)
        if not len(np.unique(masked_y_test)) < 2:
            self.logger.info("Masked Test Label Distribution:")
            ModelAnalyzer().print_label_distribution(masked_y_test, logger = self.logger)
        
        startTime = datetime.now()
        num_leaves = self.params['Akin_num_leaves']
        num_boost_round = self.params['Akin_num_boost_round']
        self.logger.info(f"LGBM: num_leaves: {num_leaves}, num_boost_round: {num_boost_round}")
        #lgbmPostOptuna, _ = self.__run_OptunaOnFiltered(masked_X_train, masked_y_train, masked_X_val, masked_y_val, masked_X_test, enablePrint=True)
        #lgbmInstance = self.__run_LGBM(masked_X_train, masked_y_train, masked_X_val, masked_y_val, masked_sample_weights_train)
        lgbmInstance = self.__run_LGBM_reg(
            masked_X_train, 
            masked_y_train_reg, 
            masked_X_val, 
            masked_y_val_reg, 
            masked_sample_weights_train,
            num_leaves = num_leaves, 
            num_boost_round = num_boost_round
        )
        endTime = datetime.now()
        self.logger.info(f"LGBM completed in {endTime - startTime}.")
        
        masked_colnames = np.array(self.featureColumnNames)[mask_features]
        ModelAnalyzer().print_feature_importance_LGBM(lgbmInstance, masked_colnames, 10, logger = self.logger)
        
        masked_y_pred_train_reg = lgbmInstance.predict(masked_X_train, num_iteration=lgbmInstance.best_iteration)
        
        masked_y_pred_val_reg = lgbmInstance.predict(masked_X_val, num_iteration=lgbmInstance.best_iteration)
        
        masked_y_pred_test_reg = lgbmInstance.predict(masked_X_test, num_iteration=lgbmInstance.best_iteration)
        
        self.logger.info("Predicted Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_train_reg > 0.05, logger = self.logger)
        self.logger.info("Predicted Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_val_reg > 0.05, logger = self.logger)
        self.logger.info("Predicted Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_test_reg > 0.05, logger = self.logger)
        
        self.logger.info("Training Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_train_reg > 0.05, masked_y_train_reg > 0.05, None, logger = self.logger)
        self.logger.info("Validation Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_val_reg > 0.05, masked_y_val_reg > 0.05, None, logger = self.logger)
        self.logger.info("Testing Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_test_reg > 0.05, masked_y_test_reg > 0.05, None, logger = self.logger)
        
        # Top m highest 
        m = self.params['Akin_top_highest']
        top_m_indices = np.flip(np.argsort(masked_y_pred_test_reg)[-m:])
        selected_true_values_reg = masked_y_test_reg[top_m_indices]
        accuracy_top_m_above_5 = np.mean(selected_true_values_reg > 0.05)
        self.logger.info(f"Accuracy of top {m} to be over 5% in test set: {accuracy_top_m_above_5:.2%}")
        self.logger.info(f"Mean value of top {m}: {np.mean(selected_true_values_reg)}")
        self.logger.info(f"Min value of top {m}: {np.min(selected_true_values_reg)}")
        self.logger.info(f"Max value of top {m}: {np.max(selected_true_values_reg)}")
        
        selected_stocks = self.meta_test[mask_test][top_m_indices]
        for i, stock in enumerate(selected_stocks[:,0]):
            self.logger.info(f"Stock: {stock}, Date: {selected_stocks[i,1]}")
            self.logger.info(f"    prediction: {masked_y_pred_test_reg[top_m_indices[i]]}")
            #check whether stock is self.__assets
            if stock not in self.__assets:
                self.logger.info(f"    Stock {stock} is not in the assets list.")
                continue
            aidx = DPl(self.__assets[stock].shareprice).getNextLowerOrEqualIndex(self.test_date)
            self.logger.info(f"    start price: {self.__assets[stock].shareprice['Close'].item(aidx)}")
            self.logger.info(f"    end price: {self.__assets[stock].shareprice['Close'].item(aidx+self.idxAfterPrediction)}")
            self.logger.info(f"    ratio: {self.__assets[stock].shareprice['Close'].item(aidx+self.idxAfterPrediction) / self.__assets[stock].shareprice['Close'].item(aidx)}")
        
        return np.mean(selected_true_values_reg)
                
    def predict(self):
        if not self.dataIsPrepared:
            raise ValueError("Data is not prepared. Please run prepareData() first.")
        
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        startTime = datetime.now()
        nTrain = self.X_train.shape[0]
        nTest = self.X_test.shape[0]
        
        test_quantil = self.params['Akin_test_quantile']
        feature_max = self.params['Akin_feature_max']
        itersteps = self.params['Akin_itersteps']
        num_leaves = self.params['Akin_pre_num_leaves']
        num_boost_round = self.params['Akin_pre_num_boost_round']
        weight_truncation = self.params['Akin_pre_weight_truncation']
        
        self.logger.info(f"num_leaves: {num_leaves}")
        self.logger.info(f"num_boost_round: {num_boost_round}")
        self.logger.info(f"weight_truncation: {weight_truncation}")
        self.logger.info(f"test_quantil: {test_quantil}")
        self.logger.info(f"feature_max: {feature_max}")
        self.logger.info(f"iterSteps: {itersteps}")
        mask_train, mask_test, mask_features, sample_weights = self.establishMasks(
            q_test=test_quantil, 
            feature_max=feature_max, 
            iterSteps=itersteps,
            num_leaves = num_leaves,
            num_boost_round = num_boost_round,
            weight_truncation = weight_truncation,
        )        
        endTime = datetime.now()
        self.logger.info(f"Masking completed in {endTime - startTime}.")
        
        self.logger.info(f"Training samples selected: {mask_train.sum()} out of {nTrain}")
        self.logger.info(f"Test samples selected: {mask_test.sum()} out of {nTest}")
        self.logger.info(f"Features selected: {mask_features.sum()} out of {self.X_test.shape[1]}")
        
        #Establish validation set
        # Random subsample for validation set
        mask_val = np.random.rand(np.sum(mask_train)) < 0.05
        if np.sum(mask_val) < 2:  
            ValueError("Not enough data points to establish validation set")
            
        self.X_val = self.X_train[mask_train][mask_val]
        self.y_val = self.y_train[mask_train][mask_val]
        self.y_val_reg = self.y_train_timeseries[mask_train][mask_val]
        
        masked_X_train = self.X_train[mask_train][~mask_val][:, mask_features]
        masked_y_train = self.y_train[mask_train][~mask_val]
        masked_y_train_reg = self.y_train_timeseries[mask_train][~mask_val]
        masked_X_val = self.X_val[:, mask_features]
        masked_y_val = self.y_val
        masked_y_val_reg = self.y_val_reg
        masked_X_test = self.X_test[mask_test][:, mask_features]
        
        masked_sample_weights_train = sample_weights[mask_train][~mask_val]
            
        self.logger.info(f"Number of features:  {len(self.featureColumnNames)}")
        self.logger.info("Overall Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_train, logger = self.logger)
        self.logger.info("Overall Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_val, logger = self.logger)
        self.logger.info("Masked Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_train, logger = self.logger)
        self.logger.info("Masked Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_val, logger = self.logger)
        
        startTime = datetime.now()
        num_leaves = self.params['Akin_num_leaves']
        num_boost_round = self.params['Akin_num_boost_round']
        self.logger.info(f"LGBM: num_leaves: {num_leaves}, num_boost_round: {num_boost_round}")
        lgbmInstance = self.__run_LGBM_reg(
            masked_X_train, 
            masked_y_train_reg, 
            masked_X_val, 
            masked_y_val_reg, 
            masked_sample_weights_train,
            num_leaves = num_leaves, 
            num_boost_round = num_boost_round
        )
        endTime = datetime.now()
        self.logger.info(f"LGBM completed in {endTime - startTime}.")
        
        masked_colnames = np.array(self.featureColumnNames)[mask_features]
        ModelAnalyzer().print_feature_importance_LGBM(lgbmInstance, masked_colnames, 10, logger = self.logger)
        
        masked_y_pred_train_reg = lgbmInstance.predict(masked_X_train, num_iteration=lgbmInstance.best_iteration)
        
        masked_y_pred_val_reg = lgbmInstance.predict(masked_X_val, num_iteration=lgbmInstance.best_iteration)
        
        masked_y_pred_test_reg = lgbmInstance.predict(masked_X_test, num_iteration=lgbmInstance.best_iteration)
        
        self.logger.info("Predicted Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_train_reg > 0.05, logger = self.logger)
        self.logger.info("Predicted Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_val_reg > 0.05, logger = self.logger)
        self.logger.info("Predicted Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_test_reg > 0.05, logger = self.logger)
        
        self.logger.info("Training Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_train_reg > 0.05, masked_y_train_reg > 0.05, None, logger = self.logger)
        self.logger.info("Validation Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_val_reg > 0.05, masked_y_val_reg > 0.05, None, logger = self.logger)
        self.logger.info("")
        
        # Top m highest
        m = self.params['Akin_top_highest']
        top_m_indices = np.argsort(masked_y_pred_test_reg)[-m:][::-1]
        selected_stocks = self.meta_test[mask_test][top_m_indices]
        for i, stock in enumerate(selected_stocks[:,0]):
            self.logger.info(f"Stock: {stock}, prediction: {masked_y_pred_test_reg[top_m_indices[i]]}")