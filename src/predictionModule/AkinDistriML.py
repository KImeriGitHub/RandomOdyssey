import numpy as np
import pandas as pd
import polars as pl
import bisect
from typing import Dict, List
import lightgbm as lgb
import optuna
import shap
import warnings
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

    def __init__(self, assets: Dict[str, AssetDataPolars], 
                 test_date: pd.Timestamp,
                 params: dict = None,
                 gatherTestResults: bool = False):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        
        self.lagList = [1,2,3,5,7,10,13,16,21,45,60,102,200,255,290]
        self.featureColumnNames = []
        self.gatherTestResults = gatherTestResults
        self.test_date = test_date
        
        trainingInterval_days = 365*4+60
        testInterval_days = 2
        self.print_OptunaBars = True
        self.testDates = pd.date_range(self.test_date - pd.Timedelta(days=testInterval_days), self.test_date, freq='B')
        
        if not self.testDates[-1] == self.test_date:
            print("test_start_date is not a business day. Correcting to first business day in assets before test_start_date.")
            asset: AssetDataPolars = self.__assets[next(iter(self.__assets))]
            dateIdx = DPl(asset.shareprice).getNextLowerOrEqualIndex(self.test_date)
            self.test_date = pd.Timestamp(asset.shareprice["Date"].item(dateIdx))
            self.testDates = pd.date_range(self.test_date - pd.Timedelta(days=testInterval_days), self.test_date, freq='B')
        
        exampleTicker = next(iter(self.__assets))
        aidx = DPl(self.__assets[exampleTicker].shareprice).getNextLowerOrEqualIndex(self.testDates[0])
        aidx_m = aidx - self.params['idxAfterPrediction']
        train_end_date = self.__assets[exampleTicker].shareprice["Date"].item(aidx_m)
        train_start_date = train_end_date - pd.Timedelta(days=trainingInterval_days)
        self.trainDates = pd.date_range(train_start_date, train_end_date, freq='B')
        
        assert self.__assets[exampleTicker].shareprice["Date"].last() >= max(self.testDates)
        
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
        
        curAdjPrice:float = asset.adjClosePrice["AdjClose"].item(aidx)
        futureMeanPrice = asset.adjClosePrice["AdjClose"].item(aidx + self.idxAfterPrediction)
        futureMaxPrice = asset.adjClosePrice["AdjClose"].slice(aidx+1, self.idxAfterPrediction).to_numpy().max()

        futurePriceScaled = futureMaxPrice/curAdjPrice
        
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

            print(f"Processing asset: {asset.ticker}. Processed {processedCounter} out of {len(self.__assets)}.")
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
                    print("Warning! Asset History does not span far enough for features.")
                    continue
                if len(asset.shareprice["Date"]) <= aidx + self.idxAfterPrediction:
                    print(f"Asset {ticker} does not have enough data to calculate target on date {date}.")
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
                    print("Warning! Asset History does not span far enough for features.")
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
            print("  Best Value: ", study.best_value)
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")

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
            #'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
            #'feature_fraction_bynode': study.best_trial.params['feature_fraction_bynode'],
            #'feature_fraction': study.best_trial.params['feature_fraction'],
            'num_leaves': 2048//intscaler, #study.best_trial.params['num_leaves'],
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
            'boost_from_average': True,
            'objective': 'quantile',
            'alpha': 0.85,
            'metric': 'quantile',  # NOTE: the string 'rsme' is not recognized, v 4.5.0
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'early_stopping_rounds': num_boost_round//10,
            'feature_fraction': 1.0,
            'num_leaves': num_leaves, 
            'max_depth': int((np.log2(num_leaves) // 1) * 2),
            'random_state': 41,
        }   
        def print_eval_after_100(env):
            if env.iteration % 10 == 0 or env.iteration == num_boost_round:
                results = [
                    f"{data_name}'s {eval_name}: {result}"
                    for data_name, eval_name, result, _ in env.evaluation_result_list
                ]
                print(f"Iteration {env.iteration}: " + ", ".join(results))
        gbm = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=num_boost_round,
            callbacks=[print_eval_after_100]
        )   
        return gbm
    
    def __get_ksDis(self, mask_train, mask_test, mask_features):
        weightedFeatures = self.__mask_weightedFeatures()
        features_w = weightedFeatures & mask_features
        
        metric_distrEquality = np.zeros(features_w.sum())
        quantile_points = np.linspace(0.01, 0.99, 100)
        qIndices_train = np.array(quantile_points * mask_train.sum()).astype(int)
        qIndices_test = np.array(quantile_points * mask_test.sum()).astype(int)
        
        train_sorted = np.sort(self.X_train[mask_train][:, features_w], axis=0)
        test_sorted = np.sort(self.X_test[mask_test][:, features_w], axis=0)
        train_quantiles = train_sorted[qIndices_train]
        test_quantiles = test_sorted[qIndices_test]
        metric_distrEquality = np.mean(np.abs(train_quantiles - test_quantiles), axis=0)
            
        print("  Train-Test Distri Equality: ", np.quantile(metric_distrEquality, 0.9))
        
        res = np.zeros(len(mask_features), dtype=float)
        res[features_w] = metric_distrEquality
        
        del train_sorted, test_sorted, train_quantiles, test_quantiles # saving on RAM
        
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
            fImp,
            n_trunc,
            n_bin = 20
        ):

        ksdis = self.__get_ksDis(mask_train, mask_test, mask_features)
        
        mask_colToAssimilate = np.char.find(self.featureColumnNames, "FeatureTA") >= 0
        mask_colToAssimilate = mask_colToAssimilate | np.char.find(self.featureColumnNames, "MathFeature") >= 0
        mask_colToAssimilate = mask_colToAssimilate | np.char.find(self.featureColumnNames, "Fourier") >= 0
        
        ksdis_colToAssimilate = ksdis
        ksdis_colToAssimilate[~mask_colToAssimilate] = 0
        
        ksdis_argsort = np.argsort(ksdis)
        ksdis_argsort_tc = np.flip(ksdis_argsort[-n_trunc:])
        
        match_factor_arr = fImp[ksdis_argsort_tc] * ksdis[ksdis_argsort_tc]
        match_factor_arr = np.maximum(match_factor_arr, np.zeros_like(match_factor_arr))
        if np.sum(match_factor_arr) < 1e-12:
            match_factor_arr = ksdis[ksdis_argsort_tc]
        
        for i in range(len(match_factor_arr)):
            print(f"  Feature {i}: {self.featureColumnNames[ksdis_argsort_tc[i]]}")
        
        sample_weights = DistributionTools.establishMatchingWeight(
            self.X_train[mask_train][:, ksdis_argsort_tc],
            self.X_test[mask_test][:, ksdis_argsort_tc],
            match_factor_arr,
            n_bin = n_bin
        )
        sample_weights /= np.sum(mask_train) / np.sum(sample_weights)
        print("  Zeros Weight Ratio: ", np.sum(sample_weights < 1e-6) / len(sample_weights))
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
            return mask_test
        
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
            'objective': 'quantile',
            'alpha': 0.85,
            'metric': 'quantile',
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'early_stopping_rounds': num_boost_round//10,
            'feature_fraction': np.max([0.1, min(feature_max / mask_features.sum(), 1.0)]),
            'num_leaves': num_leaves,
            'max_depth': int((np.log2(num_leaves) // 1) * 2),
            'random_state': 41,
        }
        
        def print_eval_after_100(env):
            if env.iteration % 10 == 0 or env.iteration == num_boost_round:
                results = [
                    f"{data_name}'s {eval_name}: {result}"
                    for data_name, eval_name, result, _ in env.evaluation_result_list
                ]
                print(f"Iteration {env.iteration}: " + ", ".join(results))
        
        gbm = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=[print_eval_after_100]
        )
        
        if self.gatherTestResults:
            print("  Mask Testing Best Iteration: ", gbm.best_iteration)
            y_pred = gbm.predict(self.X_test[mask_test][:, mask_features], num_iteration=gbm.best_iteration)
            print("  Accuracy at Test Mask: ", np.abs(self.y_test_timeseries[mask_test]-y_pred).mean()) 
        
        return gbm
        
    def establishMasks(self, 
            q_test: float, 
            feature_max:int, 
            iterSteps: int,
            num_leaves:int = 256,
            num_boost_round:int = 5000,
            weight_truncation: int = 5,
            n_bin = 15
        ):
        mask_train = np.ones(self.X_train.shape[0], dtype=bool) #self.__establishMask_rmOutliers(self.X_train_norm, rm_ratio)
        mask_test = np.ones(self.X_test.shape[0], dtype=bool) #self.__establishMask_rmOutliers(self.X_test_norm, rm_ratio)
        mask_features = np.ones(self.X_train.shape[1], dtype=bool)
        sample_weights = np.ones(self.X_train.shape[0], dtype=float)
        feature_importances = np.ones(self.X_train.shape[1], dtype=float)
        
        p_test = q_test ** (1 / iterSteps)
        q_features = feature_max / self.X_train.shape[1]
        p_features = q_features ** (1 / iterSteps)
        
        for i in range(iterSteps):
            print(f"Establish Mask: Step {i+1}/{iterSteps}.")
            startTime_loop = datetime.now()

            # Establish Weights
            sample_weights_loop = self.__establishWeights(
                mask_train = mask_train, 
                mask_test = mask_test,
                mask_features = mask_features,
                fImp = feature_importances,
                n_trunc = weight_truncation,
                n_bin = n_bin
            )
            sample_weights[mask_train] = sample_weights_loop
            
            # Estblish LGBM Instance
            lgbmInstance = self.__establishLGBMInstance(
                mask_train = mask_train, 
                mask_test = mask_test,
                mask_features = mask_features,
                feature_max = feature_max,
                sample_weights = sample_weights[mask_train],
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
            self.__get_ksDis(
                mask_train = mask_train, 
                mask_test = mask_test,
                mask_features = mask_features,
            ) 
            print(f"  Time elapsed: {endTime_loop - startTime_loop}.")
            print("  Masked Training Label Distribution:")
            ModelAnalyzer().print_label_distribution(self.y_train[mask_train])
            if self.gatherTestResults:
                print("  Masked Test Label Distribution:")
                ModelAnalyzer().print_label_distribution(self.y_test[mask_test])
                
        # End of loop Establish Weights
        print("End of Loop computation.")
        sample_weights_loop = self.__establishWeights(
            mask_train = mask_train, 
            mask_test = mask_test,
            mask_features = mask_features,
            fImp = feature_importances,
            n_trunc = weight_truncation,
            n_bin = n_bin
        )
        sample_weights[mask_train] = sample_weights_loop
        
        return mask_train, mask_test, mask_features, sample_weights
    
    def analyze(self):
        if not self.gatherTestResults:
            raise ValueError("evaluateTestResults is set to False. Cannot analyze per filter.")
        if not self.dataIsPrepared:
            raise ValueError("Data is not prepared. Please run prepareData() first.")
        
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        nTrain = self.X_train.shape[0]
        nTest = self.X_test.shape[0]
        
        test_quantil = self.params['Akin_test_quantile']
        feature_max = self.params['Akin_feature_max']
        itersteps = self.params['Akin_itersteps']
        num_leaves = self.params['Akin_pre_num_leaves']
        num_boost_round = self.params['Akin_pre_num_boost_round']
        weight_truncation = self.params['Akin_pre_weight_truncation']
        
        print(f"num_leaves: {num_leaves}")
        print(f"num_boost_round: {num_boost_round}")
        print(f"weight_truncation: {weight_truncation}")
        print(f"test_quantil: {test_quantil}")
        print(f"feature_max: {feature_max}")
        print(f"iterSteps: {itersteps}")
        
        startTime = datetime.now()
        mask_train, mask_test, mask_features, sample_weights = self.establishMasks(
            q_test=test_quantil, 
            feature_max=feature_max, 
            iterSteps=itersteps,
            num_leaves = num_leaves,
            num_boost_round = num_boost_round,
            weight_truncation = weight_truncation,
        )        
        endTime = datetime.now()
        print(f"Masking completed in {endTime - startTime}.")
        
        print(f"Training samples selected: {mask_train.sum()} out of {nTrain}")
        print(f"Test samples selected: {mask_test.sum()} out of {nTest}")
        print(f"Features selected: {mask_features.sum()} out of {self.X_test.shape[1]}")
        
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
            
        print("Number of features: ", len(self.featureColumnNames))
        print("Overall Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_train)
        print("Overall Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_val)
        print("Overall Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_test)
            
        if (
            len(np.unique(masked_y_test)) < 2
            or len(np.unique(masked_y_val)) < 2
            or len(np.unique(masked_y_train)) < 2
        ):
            print("STOPPED! Due to insufficient classes in masked.")
            print("Classes in masked training set: ", np.unique(masked_y_train))
            print("Classes in masked validation set: ", np.unique(masked_y_val))
            print("Classes in test set: ", np.unique(masked_y_test))
            return
        
        print("Masked Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_train)
        print("Masked Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_val)
        print("Masked Test Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_test)
        
        startTime = datetime.now()
        num_leaves = self.params['Akin_num_leaves']
        num_boost_round = self.params['Akin_num_boost_round']
        print(f"LGBM: num_leaves: {num_leaves}, num_boost_round: {num_boost_round}")
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
        print(f"LGBM completed in {endTime - startTime}.")
        
        masked_colnames = np.array(self.featureColumnNames)[mask_features]
        ModelAnalyzer().print_feature_importance_LGBM(lgbmInstance, masked_colnames, 10)
        
        masked_y_pred_train_reg = lgbmInstance.predict(masked_X_train, num_iteration=lgbmInstance.best_iteration)
        
        masked_y_pred_val_reg = lgbmInstance.predict(masked_X_val, num_iteration=lgbmInstance.best_iteration)
        
        masked_y_pred_test_reg = lgbmInstance.predict(masked_X_test, num_iteration=lgbmInstance.best_iteration)
        
        print("Predicted Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_train_reg > 0.05)
        print("Predicted Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_val_reg > 0.05)
        print("Predicted Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_test_reg > 0.05)
        
        print("Training Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_train_reg > 0.05, masked_y_train_reg > 0.05, None)
        print("Validation Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_val_reg > 0.05, masked_y_val_reg > 0.05, None)
        print("Testing Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_test_reg > 0.05, masked_y_test_reg > 0.05, None)
        
        # Top m highest 
        m = self.params['Akin_top_highest']
        top_m_indices = np.argsort(masked_y_pred_test_reg)[-m:][::-1]
        selected_true_values = masked_y_test[top_m_indices]
        selected_true_values_reg = masked_y_test_reg[top_m_indices]
        accuracy_top_m_above_5 = np.mean(selected_true_values > 0.05)
        print(f"Accuracy of top {m} to be over 5% in test set: {accuracy_top_m_above_5:.2%}")
        print(f"Mean value of top {m}: {np.mean(selected_true_values_reg)}")
        print(f"Min value of top {m}: {np.min(selected_true_values_reg)}")
        print(f"Max value of top {m}: {np.max(selected_true_values_reg)}")
        
        selected_stocks = self.meta_test[top_m_indices]
        for i, stock in enumerate(selected_stocks[:,0]):
            print(f"Stock: {stock}")
            print(f"    prediction: {masked_y_pred_test_reg[top_m_indices[i]]}")
            aidx = DPl(self.__assets[stock].shareprice).getNextLowerOrEqualIndex(self.test_date)
            print(f"    start price: {self.__assets[stock].shareprice['Close'].item(aidx)}")
            print(f"    end price: {self.__assets[stock].shareprice['Close'].item(aidx+self.idxAfterPrediction)}")
            print(f"    ratio: {self.__assets[stock].shareprice['Close'].item(aidx+self.idxAfterPrediction) / self.__assets[stock].shareprice['Close'].item(aidx)}")
        
        return accuracy_top_m_above_5
                
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
        
        print(f"num_leaves: {num_leaves}")
        print(f"num_boost_round: {num_boost_round}")
        print(f"weight_truncation: {weight_truncation}")
        print(f"test_quantil: {test_quantil}")
        print(f"feature_max: {feature_max}")
        print(f"iterSteps: {itersteps}")
        mask_train, mask_test, mask_features, sample_weights = self.establishMasks(
            q_test=test_quantil, 
            feature_max=feature_max, 
            iterSteps=itersteps,
            num_leaves = num_leaves,
            num_boost_round = num_boost_round,
            weight_truncation = weight_truncation,
        )        
        endTime = datetime.now()
        print(f"Masking completed in {endTime - startTime}.")
        
        print(f"Training samples selected: {mask_train.sum()} out of {nTrain}")
        print(f"Test samples selected: {mask_test.sum()} out of {nTest}")
        print(f"Features selected: {mask_features.sum()} out of {self.X_test.shape[1]}")
        
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
            
        print("Number of features: ", len(self.featureColumnNames))
        print("Overall Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_train)
        print("Overall Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_val)
        print("Masked Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_train)
        print("Masked Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_val)
        
        startTime = datetime.now()
        num_leaves = self.params['Akin_num_leaves']
        num_boost_round = self.params['Akin_num_boost_round']
        print(f"LGBM: num_leaves: {num_leaves}, num_boost_round: {num_boost_round}")
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
        print(f"LGBM completed in {endTime - startTime}.")
        
        masked_colnames = np.array(self.featureColumnNames)[mask_features]
        ModelAnalyzer().print_feature_importance_LGBM(lgbmInstance, masked_colnames, 10)
        
        masked_y_pred_train_reg = lgbmInstance.predict(masked_X_train, num_iteration=lgbmInstance.best_iteration)
        
        masked_y_pred_val_reg = lgbmInstance.predict(masked_X_val, num_iteration=lgbmInstance.best_iteration)
        
        masked_y_pred_test_reg = lgbmInstance.predict(masked_X_test, num_iteration=lgbmInstance.best_iteration)
        
        print("Predicted Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_train_reg > 0.05)
        print("Predicted Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_val_reg > 0.05)
        print("Predicted Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_test_reg > 0.05)
        
        print("Training Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_train_reg > 0.05, masked_y_train_reg > 0.05, None)
        print("Validation Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_val_reg > 0.05, masked_y_val_reg > 0.05, None)
        print()
        
        # Top m highest
        m = self.params['Akin_top_highest']
        top_m_indices = np.argsort(masked_y_pred_test_reg)[-m:][::-1]
        selected_stocks = self.meta_test[top_m_indices]
        for i, stock in enumerate(selected_stocks[:,0]):
            print(f"Stock: {stock}, prediction: {masked_y_pred_test_reg[top_m_indices[i]]}")