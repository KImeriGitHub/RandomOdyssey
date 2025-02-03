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
                 test_start_date: pd.Timestamp,
                 params: dict = None,
                 evaluateTestResults: bool = True):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        
        self.lagList = [1,2,3,5,7,10,13,16,21,45,60,102,200,255,290]
        self.featureColumnNames = []
        self.evaluateTestResults = evaluateTestResults
        self.test_start_date = test_start_date
        
        trainingInterval_days = 365*1+60
        testInterval_days = 5
        self.quantil: float = 0.1
        self.print_OptunaBars = True
        self.testDates = pd.date_range(self.test_start_date - pd.Timedelta(days=testInterval_days), self.test_start_date, freq='B')
        
        if not self.testDates[-1] == self.test_start_date:
            print("test_start_date is not a business day. Correcting to first business day in assets before test_start_date.")
            asset: AssetDataPolars = self.__assets[next(iter(self.__assets))]
            dateIdx = DPl(asset.shareprice).getNextLowerOrEqualIndex(self.test_start_date)
            self.test_start_date = pd.Timestamp(asset.shareprice["Date"].item(dateIdx))
        
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
        self.metadata['Subset_params'] = self.params
        
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
                
                if self.evaluateTestResults:
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
        intscaler = 16
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
    
    def __run_LGBM_reg(self, mask_X_Train, mask_y_Train_reg, mask_X_val, mask_y_val_reg, sample_weights):
        intscaler = 16
        best_params  = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'objective': 'regression',
            'metric': 'rsme',
            #'early_stopping_rounds': 2000//intscaler,
            'num_boost_round': 40000//intscaler,
            #'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
            #'feature_fraction_bynode': study.best_trial.params['feature_fraction_bynode'],
            #'feature_fraction': study.best_trial.params['feature_fraction'],
            'num_leaves': 2048//intscaler, #study.best_trial.params['num_leaves'],
            'random_state': 41,
        }
        lgbmInstance = lgb.LGBMRegressor(**best_params)
        #lgbmInstance.fit(X_final, y_final, sample_weight=w_final)        
        lgbmInstance.fit(mask_X_Train, mask_y_Train_reg,
                         eval_set=[(mask_X_val, mask_y_val_reg)],
                         sample_weight=sample_weights)       
        return lgbmInstance
        
    """
    # Compute distances
    def rowwise_dist(A, B, batch_size=20):
        # A: shape (nA, n_features), B: shape (nB, n_features)
        # Returns shape (nA, n_features)
        
        # Distance to the distribution of B in every feature. Then take sqrt and then mean.
        #return np.mean(np.sqrt(np.abs(A[:, None, :] - B)), axis=1)
        # TOO RAM INTENSIVE
        
        nA, n_features = A.shape
        nB = B.shape[0]
        result = np.empty((nA, n_features), dtype=A.dtype)

        for start in range(0, nA, batch_size):
            end = start + batch_size
            A_batch = A[start:end]  # Shape: (batch_size, n_features)
            # Compute distances for the batch
            diff = np.abs(A_batch[:, None, :] - B)  # Shape: (batch_size, nB, n_features)
            distances = np.mean(np.sqrt(diff), axis=1)  # Shape: (batch_size, n_features)
            result[start:end] = distances

        return result
    """
    
    """
    def mask_subset_in_superset(A, B):
        #
        #Returns a boolean array of size N with L True values, selecting the subset
        #specified by B from the True entries in A.
        #
        original_mask = np.array(A, dtype=bool)
        subset_mask = np.array(B, dtype=bool)
        idx = np.where(original_mask)[0]
        out = np.zeros_like(original_mask, dtype=bool)
        out[idx[subset_mask]] = True
        return out
    """
    
    def __get_ksDis(self, Xtr, Xte, fImp, mask_features):
        fTrain = Xtr.shape[1]
        fTest = Xte.shape[1]
        
        assert fTrain == fTest == len(fImp), "Feature dimensions are not equal."
        assert np.all(fImp>=0), "Feature importances are not non-negative."
        
        weightedFeatures = self.__mask_weightedFeatures()
        features_w = weightedFeatures[mask_features]
        
        Xtr_masked = Xtr[:, features_w]
        Xte_masked = Xte[:, features_w]
        fImp_masked = fImp[features_w]
        
        metric_distrEquality_train = np.zeros(len(fImp_masked))
        quantile_points = np.linspace(0.01, 0.99, 100)
        qIndices_train = np.array(quantile_points * Xtr.shape[0]).astype(int)
        qIndices_test = np.array(quantile_points * Xte.shape[0]).astype(int)
        for i in range(Xtr_masked.shape[1]):
            train_sorted = np.sort(Xtr_masked[:, i])
            test_sorted = np.sort(Xte_masked[:, i])
            
            train_quantiles = train_sorted[qIndices_train]
            test_quantiles = test_sorted[qIndices_test]
            
            metric_distrEquality_train[i] = np.mean(np.abs(train_quantiles - test_quantiles))
            
        print("  Train-Test Distri Equality: ", np.quantile(metric_distrEquality_train, 0.9))
        
        res = np.zeros(fTest)
        res[features_w] = metric_distrEquality_train
        return res
        
    def __mask_weightedFeatures(self):
        feature_names = np.array(self.featureColumnNames).astype(str)
        patterns = [
            'Seasonal', 
            'Category', 
            'daysToReport', 
            '_rank', '_Rank', '_RANK',
            ]
        mask = np.ones(self.X_train.shape[1], dtype=bool)
        for i in range(self.X_train.shape[1]):
            if any([pattern in feature_names[i] for pattern in patterns]):
                mask[i] = False
                continue
            
            min_train = np.quantile(self.X_train_norm[:,i],0.05)
            max_train = np.quantile(self.X_train_norm[:,i],0.95)
            min_test = np.quantile(self.X_test_norm[:,i],0.01)
            max_test = np.quantile(self.X_test_norm[:,i],0.99)
            
            if max_train - min_train < 1e-4:  # TODO: There is one such thing. To investigate! Perhaps the year.
                mask[i] = False
            
            if max_train - min_train > 4 * (max_test - min_test):
                mask[i] = False
            
        return mask
        
    def __establishMask_rmOutliers(self, test_norm: np.array, rm_ratio: float):
        clf = IsolationForest(contamination=rm_ratio, random_state=41)
        clf.fit(test_norm)
        predictions = clf.predict(test_norm)
        return  (predictions > 0)
        
    #def __establishAkinMask_TrainVal(self, ptr:float, pv:float, Xtr, Xv, Xte, mask_features):
    #    fTrain = Xtr.shape[1]
    #    fVal = Xv.shape[1]
    #    fTest = Xte.shape[1]
    #    
    #    assert fTrain == fVal == fTest, "Feature dimensions are not equal."
    #    
    #    if ptr >= 1 - 1e-12 and pv >= 1 - 1e-12:
    #        return np.ones(len(Xtr), dtype=bool), np.ones(len(Xv), dtype=bool)
    #    
    #    weightedFeatures = self.__mask_weightedFeatures()
    #    features_w = weightedFeatures[mask_features]
    #    
    #    Xtr_masked = Xtr[:, features_w]
    #    Xv_masked = Xv[:, features_w]
    #    Xte_masked = Xte[:, features_w]
    #    
    #    ksDis_train = self.__get_ksDis_vec(Xtr_masked, Xte_masked)
    #    n_features = Xtr_masked.shape[1]
    #    interp_num = 50
    #
    #    # Precompute per-feature mins and maxes
    #    mins_train, maxs_train = np.min(Xtr_masked, axis=0), np.max(Xtr_masked, axis=0)
    #    mins_val, maxs_val = np.min(Xv_masked, axis=0), np.max(Xv_masked, axis=0)
    #    mins_test, maxs_test = np.min(Xte_masked, axis=0), np.max(Xte_masked, axis=0)
    #
    #    # Random subsample for training KDE
    #    mask_kde = np.random.rand(Xtr_masked.shape[0]) < 0.1
    #    if np.sum(mask_kde) < 2:  # ensure at least two points
    #        mask_kde[:] = True
    #
    #    def process_feature(i):
    #        pts_tr = np.linspace(mins_train[i], maxs_train[i], interp_num)
    #        pts_val = np.linspace(mins_val[i], maxs_val[i], interp_num)
    #        pts_test = np.linspace(mins_test[i], maxs_test[i], interp_num)
    #
    #        kde_tr = stats.gaussian_kde(Xtr_masked[mask_kde, i])
    #        kde_val = stats.gaussian_kde(Xv_masked[:, i])
    #        kde_test = stats.gaussian_kde(Xte_masked[:, i])
    #
    #        kde_tr_vals = kde_tr(pts_tr)
    #        kde_val_vals = kde_val(pts_val)
    #        kde_test_vals = kde_test(pts_test)
    #
    #        interp_tr_at_tr = np.interp(Xtr_masked[:, i], pts_tr, kde_tr_vals)
    #        interp_tr_at_test = np.interp(Xtr_masked[:, i], pts_test, kde_test_vals)
    #        interp_val_at_val = np.interp(Xv_masked[:, i], pts_val, kde_val_vals)
    #        interp_val_at_test = np.interp(Xv_masked[:, i], pts_test, kde_test_vals)
    #
    #        contr_tr = ksDis_train[i] * (interp_tr_at_tr - interp_tr_at_test)
    #        return contr_tr
    #
    #    # Parallelize feature processing
    #    results = Parallel(n_jobs=-1)(delayed(process_feature)(i) for i in range(n_features))
    #
    #    weightedSum_train = np.sum([r[0] for r in results], axis=0)
    #    weightedSum_val = np.sum([r[1] for r in results], axis=0)
    #
    #    cutoff_train = np.quantile(weightedSum_train, ptr)
    #    mask_train = weightedSum_train <= cutoff_train
    #
    #    cutoff_val = np.quantile(weightedSum_val, pv)
    #    mask_val = weightedSum_val <= cutoff_val
    #
    #    return mask_train, mask_val
        
    def __establishWeights(self, Xtr, Xte, mask_features, fImp):
        fTrain = Xtr.shape[1]
        fTest = Xte.shape[1]
        nTrain = Xtr.shape[0]
        nTest = Xte.shape[0]
        
        assert fTrain == fTest == len(fImp), "Feature dimensions are not equal."
        assert np.all(fImp>=0), "Feature importances are not non-negative."
        
        weightedFeatures = self.__mask_weightedFeatures()
        features_w = weightedFeatures[mask_features]
        ksdis = self.__get_ksDis(Xtr, Xte, fImp, mask_features)
        
        Xtr_masked = Xtr[:, features_w]
        Xte_masked = Xte[:, features_w]
        fImp_masked = fImp[features_w]
        fImp_masked_cs = np.cumsum(fImp_masked)
        
        ksdis_masked = ksdis[features_w]
        ksdis_masked_cs = np.cumsum(ksdis_masked)
        
        sample_weights = np.ones(nTrain, dtype=float)
        window = np.ones(nTest) / nTest
        ksdis_argmax = np.argmax(ksdis_masked)
        for i in range(Xtr_masked.shape[1]):
            if i != ksdis_argmax:
                continue
            samTr = Xtr_masked[:, i]
            samTe = Xte_masked[:, i]
            argsort_samTr = np.argsort(samTr)
            argsort_samTe = np.argsort(samTe)
            sort_samTr = samTr[argsort_samTr]
            sort_samTe = samTe[argsort_samTe]
            
            lo = np.searchsorted(sort_samTe, sort_samTr[:-1], side="right")
            hi = np.searchsorted(sort_samTe, sort_samTr[1:], side="left")
            weights_sorted = hi - lo
            weights_sorted = np.convolve(weights_sorted, window, mode='same')
            weights_sorted = np.insert(weights_sorted, 0, weights_sorted[0])
            weights_sorted *= nTrain/np.sum(weights_sorted)
            
            weights = weights_sorted[np.argsort(argsort_samTr)]
            
            sample_weights = weights
            
            #if i == 0:
            #    sample_weights = weights
            #    continue
            #if fImp_masked_cs[i-1] < 1e-12:
            #    sample_weights = weights
            #    continue
            
            #tau = ksdis_masked[i] / ksdis_masked_cs[i]
            #sample_weights = (1-tau) * sample_weights + tau * weights
            
        return sample_weights
        
    def __establishAkinMask_Features(self, p_features: float, lgbmInstance: lgb.Booster, Xtr):
        fTrain = Xtr.shape[1]
        
        if p_features >= 1 - 1e-12:
            return np.ones(fTrain, dtype=bool)
        
        # SHAP FEATURE IMPORTANCE
        #warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray")
        #explainer = shap.TreeExplainer(lgbmInstance)
        #shap_values = explainer.shap_values(Xtr)
        #feature_importances = np.abs(shap_values).mean(axis=0)
        #shap_quant = np.quantile(feature_importances, 1-p_features)
        #
        #mask = feature_importances > shap_quant
        
        # BUILD-IN LGBM FEATURE IMPORTANCE
        feature_importances = lgbmInstance.feature_importance()
        argsort_features = np.argsort(feature_importances)
        pN = int(np.ceil(p_features * len(feature_importances)))
        top_indices = argsort_features[-pN:]
        mask = np.zeros_like(feature_importances, dtype=bool)
        mask[top_indices] = True
        
        return mask, feature_importances
    
    def __establishAkinMask_Test(self, p_test, lgbmInstance: lgb.Booster, Xte):
        if p_test >= 1 - 1e-12:
            return np.ones(len(Xte), dtype=bool)
        
        y_pred = lgbmInstance.predict(Xte)
        
        # Quantile regression to subset mask
        mask = y_pred >= np.quantile(y_pred, 1-p_test)
        return mask
    
    def __establishLGBMInstance(self, Xtr, Xte, ytr_reg, yte_reg, feature_max:int, sample_weights):
        # yte_reg to be used with caution, because of leakage
        fTrain = Xtr.shape[1]
        fTest = Xte.shape[1]
        assert fTrain == fTest, "Feature dimensions are not equal."
        
        # Random subsample for validation set
        mask_val = np.random.rand(Xtr.shape[0]) < 0.05
        if np.sum(mask_val) < 2:  
            ValueError("Not enough data points to establish validation set")
        
        train_data = lgb.Dataset(Xtr[~mask_val], label = ytr_reg[~mask_val], weight=sample_weights[~mask_val])
        test_data = lgb.Dataset(Xtr[mask_val], label = ytr_reg[mask_val], reference=train_data)

        intscaler = 16
        params = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'objective': 'quantile',
            'alpha': 0.8,
            'metric': 'quantile',
            'early_stopping_rounds': 2000//intscaler,
            'feature_fraction': np.max([0.1, min(feature_max / fTrain, 1.0)]),
            'num_leaves': 1024//intscaler,
            'max_depth': 20,
            'learning_rate': 0.05,
            'random_state': 41,
        }
    
        gbm = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=10000//intscaler,
        )
        
        print("  Mask Testing Best Iteration: ", gbm.best_iteration)
        y_pred = gbm.predict(Xte)
        print("  Accuracy at Test Mask: ", np.abs(yte_reg-y_pred).mean()) 
        
        return gbm
        
    def establishMasks(self, q_test: float, feature_max:int, iterSteps: int, rm_ratio = 0.01):
        mask_train = np.ones(self.X_train.shape[0], dtype=bool) #self.__establishMask_rmOutliers(self.X_train_norm, rm_ratio)
        mask_test = np.ones(self.X_test.shape[0], dtype=bool) #self.__establishMask_rmOutliers(self.X_test_norm, rm_ratio)
        mask_features = np.ones(self.X_train.shape[1], dtype=bool)
        sample_weights = np.ones(self.X_train.shape[0], dtype=float)
        feature_importances = np.ones(self.X_train.shape[1], dtype=float)
        
        p_test = q_test ** (1 / iterSteps)
        q_features = feature_max / self.X_test.shape[1]
        p_features = q_features ** (1 / iterSteps)
        
        for i in range(iterSteps):
            print(f"Establish Mask: Step {i+1}/{iterSteps}.")
            startTime_loop = datetime.now()

            # Establish Weights
            sample_weights_loop = self.__establishWeights(
                Xtr = self.X_train_norm[mask_train][:, mask_features], 
                Xte = self.X_test_norm[mask_test][:, mask_features],
                mask_features = mask_features,
                fImp = feature_importances[mask_features],
            )
            sample_weights[mask_train] = sample_weights_loop
            
            # Estblish LGBM Instance
            lgbmInstance = self.__establishLGBMInstance(
                Xtr = self.X_train_norm[mask_train][:, mask_features], 
                Xte = self.X_test_norm[mask_test][:, mask_features],
                ytr_reg = self.y_train_timeseries[mask_train], 
                yte_reg = self.y_test_timeseries[mask_test],
                feature_max = feature_max,
                sample_weights = sample_weights[mask_train]
            )
            
            # Establish Test Mask: quantil regression to subset mask
            mask_test_loop = self.__establishAkinMask_Test(
                p_test,
                lgbmInstance = lgbmInstance,
                Xte=self.X_test_norm[mask_test][:, mask_features],
            )
            mask_test[mask_test] = mask_test_loop
            
            # Establish Feature Mask
            mask_features_loop, feature_importances_loop = self.__establishAkinMask_Features(
                p_features, 
                lgbmInstance = lgbmInstance,
                Xtr = self.X_train_norm[mask_train][:, mask_features],
            )
            feature_importances[mask_features] = feature_importances_loop
            mask_features[mask_features] = mask_features_loop
            
            endTime_loop = datetime.now()
            self.__get_ksDis(
                Xtr = self.X_train_norm[mask_train][:, mask_features], 
                Xte = self.X_test_norm[mask_test][:, mask_features],
                fImp = feature_importances[mask_features],
                mask_features = mask_features
            ) 
            print(f"  Time elapsed: {endTime_loop - startTime_loop}.")
            print("  Masked Training Label Distribution:")
            ModelAnalyzer().print_label_distribution(self.y_train[mask_train])
            print("  Masked Test Label Distribution:")
            ModelAnalyzer().print_label_distribution(self.y_test[mask_test])
        
        return mask_train, mask_test, mask_features, sample_weights
    
    def analyze_perFilter(self):
        if not self.evaluateTestResults:
            raise ValueError("evaluateTestResults is set to False. Cannot analyze per filter.")
        if not self.dataIsPrepared:
            raise ValueError("Data is not prepared. Please run prepareData() first.")
        
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train_norm = scaler.transform(self.X_train)
        self.X_test_norm = scaler.transform(self.X_test)
        
        startTime = datetime.now()
        nTrain = self.X_train.shape[0]
        nTest = self.X_test.shape[0]
        
        train_quantil = 0.6
        test_quantil = 0.6
        feature_max = 500
        itersteps = 5
        
        mask_train, mask_test, mask_features, sample_weights = self.establishMasks(
                q_test=test_quantil, 
                feature_max=feature_max, 
                iterSteps=itersteps,
            )        
        endTime = datetime.now()
        
        #Establish validation set
        # Random subsample for validation set
        mask_val = np.random.rand(np.sum(mask_train)) < 0.05
        if np.sum(mask_val) < 2:  
            ValueError("Not enough data points to establish validation set")
            
        self.X_val = self.X_train[mask_train][mask_val]
        self.y_val = self.y_train[mask_train][mask_val]
        self.y_val_reg = self.y_train_timeseries[mask_train][mask_val]
        
        print(f"train_quantil: {train_quantil}")
        print(f"test_quantil: {test_quantil}")
        print(f"feature_max: {feature_max}")
        print(f"iterSteps: {itersteps}")
        print(f"Masking completed in {endTime - startTime}.")
        print(f"Training samples selected: {mask_train.sum()} out of {nTrain}")
        print(f"Test samples selected: {mask_test.sum()} out of {nTest}")
        print(f"Features selected: {mask_features.sum()} out of {self.X_test.shape[1]}")
        
        print("Number of features: ", len(self.featureColumnNames))
        print("Overall Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_train)
        print("Overall Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_val)
        print("Overall Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_test)
        
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
        
        #lgbmPostOptuna, _ = self.__run_OptunaOnFiltered(masked_X_train, masked_y_train, masked_X_val, masked_y_val, masked_X_test, enablePrint=True)
        #lgbmInstance = self.__run_LGBM(masked_X_train, masked_y_train, masked_X_val, masked_y_val, masked_sample_weights_train)
        lgbmInstance = self.__run_LGBM_reg(masked_X_train, masked_y_train_reg, masked_X_val, masked_y_val_reg, masked_sample_weights_train)
        
        masked_colnames = np.array(self.featureColumnNames)[mask_features]
        ModelAnalyzer().print_feature_importance_LGBM(lgbmInstance, masked_colnames, 10)
        
        masked_y_pred_train_reg = lgbmInstance.predict(masked_X_train)
        
        masked_y_pred_val_reg = lgbmInstance.predict(masked_X_val)
        
        masked_y_pred_test_reg = lgbmInstance.predict(masked_X_test)
        
        print("Predicted Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_train_reg > 0.05)
        print("Predicted Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_val_reg > 0.05)
        print("Predicted Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(masked_y_pred_test_reg > 0.05)
        
        print("Validation Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_val_reg > 0.05, masked_y_val_reg > 0.05, None)
        print("Testing Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(masked_y_pred_test_reg > 0.05, masked_y_test_reg > 0.05, None)
        
        # Top 5 highest 
        top_5_indices = np.argsort(masked_y_pred_test_reg)[-5:][::-1]
        selected_true_values = masked_y_test[top_5_indices]
        accuracy_top_5_above_50 = np.mean(selected_true_values > 0.05)
        print(f"Accuracy of top 5 to be over 5% in test set: {accuracy_top_5_above_50:.2%}")
        
        return accuracy_top_5_above_50
        

    def predict_perFilter(self, lgbModelsList: List[lgb.LGBMClassifier] = None):
        if not self.dataIsPrepared:
            raise ValueError("Data is not prepared. Please run prepareData() first.")
        
        self.establishFilterDict()
        
        params = {
            'idxLengthOneMonth': self.metadata['Subset_params']['idxLengthOneMonth'],
            'fouriercutoff': self.metadata['Subset_params']['fouriercutoff'],
            'multFactor': self.metadata['Subset_params']['multFactor'],
            'monthsHorizon': self.metadata['Subset_params']['monthsHorizon'],
            'timesteps': self.metadata['Subset_params']['timesteps'],
        }
        
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice is None or not 'AdjClose' in asset.adjClosePrice.columns:
                raise ValueError(f"Asset {ticker} does not have adjusted close price data.")
            
            featureMain = FeatureMain(
                asset, 
                self.test_start_date, 
                self.test_start_date,
                lagList = self.lagList, 
                params=params,
                enableTimeSeries = False
            )

            aidx = DPl(asset.shareprice).getNextLowerOrEqualIndex(self.test_start_date)
            features = self.getFeatures(asset, featureMain, self.test_start_date, aidx)
            
            curPrice = asset.shareprice["Close"].item(aidx)
            futurePrice = None
            if not aidx + self.idxAfterPrediction >= len(asset.adjClosePrice["AdjClose"]):
                futurePrice = asset.shareprice["Close"].item(aidx + self.idxAfterPrediction)
                
            print(f"Asset: {ticker}   Current Price: {curPrice}")
            for i, val in enumerate(self.filterColumnsTuple_names_quantilDown_quantilUp):
                filterName = val[0]
                quantil_Down = val[1]
                quantil_Up = val[2]
                
                columnToSubset = self.filterColumnsDict_namesToIndex[filterName]
                mask_quantileDown = np.quantile(self.X_test[:, columnToSubset], quantil_Down)
                mask_quantileUp = np.quantile(self.X_test[:, columnToSubset], quantil_Up)
                
                print(f"  Filter: {filterName}. Optuna Value: {self.best_values[i]}")
                if features[columnToSubset] >= mask_quantileDown and features[columnToSubset] <= mask_quantileUp:
                    lgbmModel: lgb.LGBMClassifier = self.lgbModelsList[i]
                    prediction = lgbmModel.predict([features])[0]
                    proba = lgbmModel.predict_proba([features])[0]
                    print(f"  Prediction: {prediction}, Probabilities: {proba}")
                else:
                    print(f"  Did does not pass filter {filterName}.")
                    
            if not aidx + self.idxAfterPrediction >= len(asset.adjClosePrice["AdjClose"]):
                print(f"  End Price: {futurePrice}")
                
    def predict(self, lgbModelsList: List[lgb.LGBMClassifier] = None):
        if not self.dataIsPrepared:
            raise ValueError("Data is not prepared. Please run prepareData() first.")
        
        self.establishFilterDict()
        
        params = {
            'idxLengthOneMonth': self.metadata['Subset_params']['idxLengthOneMonth'],
            'fouriercutoff': self.metadata['Subset_params']['fouriercutoff'],
            'multFactor': self.metadata['Subset_params']['multFactor'],
            'monthsHorizon': self.metadata['Subset_params']['monthsHorizon'],
            'timesteps': self.metadata['Subset_params']['timesteps'],
        }
        
        # We'll collect all predictions, probabilities, and returns in a structure for easy strategy decisions later.
        # structuredPredictionDict[ticker] = {
        #    'return': float or None,
        #    'predictions': { filterName -> int },
        #    'probabilities': { filterName -> [prob_of_class0, prob_of_class1] }
        # }
        structuredPredictionDict = {}
        
        # initialize return dictionnary per filter and per asset
        returnDict = {}
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice is None or not 'AdjClose' in asset.adjClosePrice.columns:
                raise ValueError(f"Asset {ticker} does not have adjusted close price data.")

            featureMain = FeatureMain(
                asset, 
                self.test_start_date, 
                self.test_start_date,
                lagList=self.lagList, 
                params=params,
                enableTimeSeries=False
            )

            aidx = DPl(asset.shareprice).getNextLowerOrEqualIndex(self.test_start_date)
            features = self.getFeatures(asset, featureMain, self.test_start_date, aidx)

            curPrice = asset.adjClosePrice["AdjClose"].item(aidx)
            futurePrice = None
            if (aidx + self.idxAfterPrediction) < len(asset.adjClosePrice["AdjClose"]):
                futurePrice = asset.adjClosePrice["AdjClose"].item(aidx + self.idxAfterPrediction)

            forwardReturn = futurePrice / curPrice if futurePrice else None

            predictionDict = {}
            probaDict = {}
            for i, (filterName, quantil_Down, quantil_Up) in enumerate(self.filterColumnsTuple_names_quantilDown_quantilUp):
                colIdx = self.filterColumnsDict_namesToIndex[filterName]

                # The quantiles were computed on X_test (the full test set).
                mask_quantileDown = np.quantile(self.X_test[:, colIdx], quantil_Down)
                mask_quantileUp   = np.quantile(self.X_test[:, colIdx], quantil_Up)

                # Check if features[colIdx] is within [mask_quantileDown, mask_quantileUp]
                if mask_quantileDown <= features[colIdx] and features[colIdx] <= mask_quantileUp:
                    # The asset passes the filter, so we can use the corresponding model
                    lgbmModel: lgb.LGBMClassifier = self.lgbModelsList[i]
                    if lgbmModel is None:
                        # Means we had insufficient classes during training for that filter
                        prediction = -1
                        proba = [-1, -1]
                    else:
                        # Make predictions
                        prediction = lgbmModel.predict([features])[0]
                        proba = lgbmModel.predict_proba([features])[0]
                else:
                    # Does not pass the filter, so we set placeholders
                    prediction = -1
                    proba = [-1, -1]

                predictionDict[filterName] = prediction
                probaDict[filterName] = proba

            # Store results in our structured dictionary
            structuredPredictionDict[ticker] = {
                'return': forwardReturn,
                'predictions': predictionDict,
                'probabilities': probaDict
            }

                    
        # 1) Strategy: “Model with Best Value and asset with highest probability”
        #    - We pick the filter with the highest best_value from self.best_values.
        #    - Among the assets that pass that filter, choose the one with the highest probability of class=1.
        best_filter_idx = np.argmax(self.best_values) if len(self.best_values) > 0 else None
        if best_filter_idx is not None:
            best_filter_name = self.filterColumnsTuple_names_quantilDown_quantilUp[best_filter_idx][0]
            print("Strategy: Model with Best Value and asset with highest probability")
            print(f"  Using filter: {best_filter_name} (best_value={self.best_values[best_filter_idx]:.4f})")

            best_asset = None
            best_asset_prob = -1.0
            for ticker, results in structuredPredictionDict.items():
                proba = results['probabilities'][best_filter_name]
                prediction = results['predictions'][best_filter_name]
                if proba[0] == -1:
                    # Means the asset didn't pass the filter
                    continue
                # Probability of class=1 is proba[1]
                if proba[1] > best_asset_prob:
                    best_asset_prob = proba[1]
                    best_asset = ticker

            if best_asset is not None:
                best_return = structuredPredictionDict[best_asset]['return']
                print(f"  Ticker: {best_asset}")
                print(f"  Predicted Probability of Class=1: {best_asset_prob:.4f}")
                print(f"  Forward Return (if available): {best_return}")
            else:
                print("  No asset passed this filter or best_filter_idx is invalid.")

        # 2) Strategy: “Asset with the most class 1 predictions”
        #    - For each asset, sum up all filters that gave prediction=1.
        #    - If tie, you might pick the one with highest average p(class=1).
        print("\nStrategy: Asset with the most class 1 predictions")
        best_asset_count = -1
        best_asset_list = []  # keep track of all that tie
        asset_counts = {}
        for ticker, results in structuredPredictionDict.items():
            predictions = results['predictions']
            # Count how many filters predicted class=1
            count_ones = sum(1 for pred in predictions.values() if pred == 1)
            asset_counts[ticker] = count_ones
            if count_ones > best_asset_count:
                best_asset_count = count_ones
                best_asset_list = [ticker]
            elif count_ones == best_asset_count:
                best_asset_list.append(ticker)

        if best_asset_count <= 0:
            print("  No asset has a class=1 prediction.")
        else:
            # If we have multiple assets tied, pick the one with the highest average probability
            if len(best_asset_list) > 1:
                # Tie-breaking
                best_ticker = None
                best_mean_prob = -1.0
                for t in best_asset_list:
                    # compute the average of p(class=1) among the filters that predicted 1
                    prob_sum = 0.0
                    count_1 = 0
                    for fName, pred in structuredPredictionDict[t]['predictions'].items():
                        if pred == 1:
                            p = structuredPredictionDict[t]['probabilities'][fName][1]
                            prob_sum += p
                            count_1 += 1
                    mean_prob = prob_sum / count_1 if count_1 else 0
                    if mean_prob > best_mean_prob:
                        best_mean_prob = mean_prob
                        best_ticker = t
                # Print
                print(f"  Tickers tied with {best_asset_count} votes: {best_asset_list}")
                print(f"  Tie broken by highest average p(1). Winner: {best_ticker}")
                print(f"  Return: {structuredPredictionDict[best_ticker]['return']}")
            else:
                best_ticker = best_asset_list[0]
                print(f"  Ticker with the most 1-predictions is {best_ticker} with {best_asset_count} votes.")
                print(f"  Return: {structuredPredictionDict[best_ticker]['return']}")

        # 3) Strategy: “Separately for each filter, asset with highest probability”
        #    - For each filter, pick the asset whose pass-probability for class=1 is highest (if any).
        print("\nStrategy: Separately for each filter, asset with highest probability")
        for i, (filterName, qDown, qUp) in enumerate(self.filterColumnsTuple_names_quantilDown_quantilUp):
            # If the model was None due to insufficient classes, skip
            if self.lgbModelsList[i] is None:
                print(f"  Filter: {filterName}, Skipped (insufficient classes).")
                continue

            # Among all assets, pick the one with the largest p(class=1) that is not -1.
            best_prob = -1.0
            best_ticker = None
            for ticker, results in structuredPredictionDict.items():
                proba = results['probabilities'][filterName]
                if proba[0] == -1:
                    # Asset didn't pass filter, skip
                    continue
                if proba[1] > best_prob:
                    best_prob = proba[1]
                    best_ticker = ticker

            if best_ticker is not None:
                print(f"  Filter: {filterName}")
                print(f"    Ticker: {best_ticker}")
                print(f"    Probability of class=1: {best_prob:.4f}")
                print(f"    Return: {structuredPredictionDict[best_ticker]['return']}")
            else:
                print(f"  Filter: {filterName}, no asset passed or has valid probability.")

        # Optionally, return or store structuredPredictionDict if you want to use it outside.
        return structuredPredictionDict
                
        
        