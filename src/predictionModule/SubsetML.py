import numpy as np
import pandas as pd
import polars as pl
import bisect
from typing import Dict, List
import lightgbm as lgb
import optuna
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.mathTools.TAIndicators import TAIndicators
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.predictionModule.IML import IML
from src.featureAlchemy.FeatureMain import FeatureMain
from src.predictionModule.ModelAnalyzer import ModelAnalyzer

class SubsetML(IML):
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 5,
        'multFactor': 6,
        'daysAfterPrediction': 21,
        'monthsHorizon': 13,
        'timesteps': 5,
        'classificationInterval': [0.05], 
    }

    def __init__(self, assets: Dict[str, AssetDataPolars], 
                 test_start_date: pd.Timestamp,
                 spareRatio: float = 0.5,
                 params: dict = None,
                 dataRatio: float = 0.1,
                 evaluateTestResults: bool = True):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets
        
        self.lagList = [1,2,3,5,10,21]
        self.featureColumnNames = []
        self.dataRatio = dataRatio
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.evaluateTestResults = evaluateTestResults
        self.test_start_date = test_start_date
        
        trainingInterval_days = 365+60
        testInterval_days = 10
        self.testDates = pd.date_range(self.test_start_date, self.test_start_date + pd.Timedelta(days=testInterval_days), freq='B')
        
        if not self.testDates[0] == self.test_start_date:
            print("test_start_date is not a business day. Correcting to first business day in assets before test_start_date.")
            asset: AssetDataPolars = self.__assets[next(iter(self.__assets))]
            dateIdx = DPl(asset.shareprice).getNextLowerOrEqualIndex(self.test_start_date)
            self.test_start_date = pd.Timestamp(asset.shareprice["Date"].item(dateIdx))
        
        exampleTicker = next(iter(self.__assets))
        aidx = DPl(self.__assets[exampleTicker].shareprice).getNextLowerOrEqualIndex(self.testDates[0])
        aidx_m = aidx - self.params['idxLengthOneMonth']-1
        train_end_date = self.__assets[exampleTicker].shareprice["Date"].item(aidx_m)
        train_start_date = train_end_date - pd.Timedelta(days=trainingInterval_days)
        self.trainDates = self.__calculate_dates(train_start_date, train_end_date, spareRatio)
        
        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.daysAfterPrediction = self.params['daysAfterPrediction']
        self.monthsHorizon = self.params['monthsHorizon']
        self.classificationInterval = self.params['classificationInterval']
        self.timesteps = self.params['timesteps'] 
        
        # Store parameters in metadata
        self.metadata['Subset_params'] = self.params
        
    def __calculate_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp, ratio: float):
        date_range = pd.date_range(start_date, end_date, freq='B') 
        n_samples = max(int(len(date_range) * ratio), 1)
        return pd.DatetimeIndex(np.random.choice(date_range, n_samples, replace=False))
    
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
        if (aidx + self.daysAfterPrediction) >= len(asset.adjClosePrice["AdjClose"]):
            raise ValueError("Asset does not have enough data to calculate target.")
        
        curAdjPrice:float = asset.adjClosePrice["AdjClose"].item(aidx)
        futureMeanPrice = asset.adjClosePrice["AdjClose"].item(aidx + self.daysAfterPrediction)
        futureMeanPriceScaled = futureMeanPrice/curAdjPrice
        
        target = self.getTargetClassification([futureMeanPriceScaled-1], self.classificationInterval)
        
        return target[0]

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
                if asset.shareprice["Date"].item(-1) < date + pd.Timedelta(days=self.daysAfterPrediction):
                    print(f"Asset {ticker} does not have enough data to calculate target on date {date}.")
                    continue

                features = self.getFeatures(asset, featureMain, date, aidx)
                target = self.getTarget(asset, featureMain, date, aidx)

                Xtrain.append(features)
                ytrain.append(target)

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
                    if asset.shareprice["Date"].item(-1) < date + pd.Timedelta(days=self.daysAfterPrediction):
                        raise ValueError(f"Asset {ticker} does not have enough data to calculate target on date: {date}.")
                    target = self.getTarget(asset, featureMain, date, aidx)
                    ytest.append(target)
                else:
                    ytest.append(-1)

        self.X_train = np.array(Xtrain)
        self.y_train = np.array(ytrain).astype(int)
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, 
            self.y_train, 
            test_size=0.2,
        )
        
        self.X_test = np.array(Xtest)
        self.y_test = np.array(ytest).astype(int)
        self.X_train_timeseries = np.array(XtrainPrice)
        self.y_train_timeseries = np.array(ytrainPrice)
        self.X_test_timeseries = np.array(XtestPrice)
        self.y_test_timeseries = np.array(ytestPrice)
        self.X_val_timeseries = np.array(XvalPrice)
        self.y_val_timeseries = np.array(yvalPrice)

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test)
        self.X_val, self.y_val = shuffle(self.X_val, self.y_val)
        
        self.dataIsPrepared = True

    def establishFilterDict(self):
        self.filterColumnsTuple_names_leq = [
            ("Fourier_Price_RSME", True),
            ("FinData_quar_surprise", False),
            ("MathFeature_Drawdown", False),
            #("FinData_quar_profitLoss_lag_qm1", True),
            ("FinData_quar_reportedEPS_lag_qm1", False),
            #("FinData_quar_surprise_lagquot_qm1", False),
            #("FinData_quar_grossProfit", False),
            #("FinData_quar_grossProfit_lag_qm1", False),
            #("FinData_metrics_log_forward_pe_ratio", True),
            ("FinData_metrics_log_forward_pe_ratio_lagquot_m1", True),
            #("FinData_metrics_log_trailing_pe_ratio_lag_m1", True),
            ("FinData_metrics_log_trailing_pe_ratio", True),
            ("FinData_ann_profitLoss_lag_qm1", True),
            ("FinData_ann_grossProfit", False),
            #("FinData_ann_grossProfit_lag_qm1", False),
            #("FinData_ann_reportedEPS", True),
            #("Fourier_Price_AbsCoeff_1", False),
            #("Fourier_Price_AbsCoeff_2", True),
            ("FeatureTA_trend_macd", False),
            #("FeatureTA_trend_macd_signal", True),
            ("FeatureTA_Open", False),
        ]
        
        self.filterColumnsDict_namesToIndex = {}
        for val in self.filterColumnsTuple_names_leq:
            self.filterColumnsDict_namesToIndex[val[0]] = self.featureColumnNames.index(val[0])
            
    def establishMask(self, filterName: str, is_leq: bool):
        columnToSubset = self.filterColumnsDict_namesToIndex[filterName] #np.random.randint(0, len(colNames))
        
        if is_leq:
            mask_quantile = np.quantile(self.X_test[:, columnToSubset], self.dataRatio)

            mask_train = self.X_train[:, columnToSubset] < mask_quantile
            mask_val = self.X_val[:, columnToSubset] < mask_quantile
            mask_test = self.X_test[:, columnToSubset] < mask_quantile
        else:
            mask_quantile = np.quantile(self.X_test[:, columnToSubset], 1-self.dataRatio)
            
            mask_train = self.X_train[:, columnToSubset] > mask_quantile
            mask_val = self.X_val[:, columnToSubset] > mask_quantile
            mask_test = self.X_test[:, columnToSubset] > mask_quantile
        
        return mask_train, mask_val, mask_test
    
    def __run_OptunaOnFiltered(self, mask_X_Train, mask_y_Train, mask_X_val, mask_y_val):
        def objective(trial):
            lgbm_params = {
                'verbosity': -1,
                'n_jobs': -1,
                'is_unbalance': True,
                'metric': 'binary_logloss',
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'n_estimators': 2000,
                'feature_fraction': trial.suggest_float('feature_fraction', 0.01, 0.1),
                'num_leaves': trial.suggest_categorical('num_leaves', [64, 128, 256]),
                'max_depth': 8,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            }
            # Initialize and train LGBM model
            LGBMModel = lgb.LGBMClassifier(**lgbm_params)
            LGBMModel.fit(
                mask_X_Train, mask_y_Train,
                eval_set=[(mask_X_val, mask_y_val)]
            )
            mask_y_val_pred = LGBMModel.predict(mask_X_val)
            cm:np.array = confusion_matrix(mask_y_val, mask_y_val_pred, labels=np.unique(mask_y_val))
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
            return np.sum(per_class_accuracy)

        # 3. Create a study2 object and optimize the objective function.
        #optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials = 20, timeout=60*5, show_progress_bar=False)
        print("  Best Value: ", study.best_value)
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

        lgbm_params = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'metric': 'binary_logloss',
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'n_estimators': 2000,
            'feature_fraction': study.best_trial.params['feature_fraction'],
            'num_leaves': study.best_trial.params['num_leaves'],
            'max_depth': 8,
            'learning_rate': study.best_trial.params['learning_rate'],
        }
        # Initialize and train LGBM model
        lgbmInstance = lgb.LGBMClassifier(**lgbm_params)
        lgbmInstance.fit(
            mask_X_Train, mask_y_Train,
            eval_set=[(mask_X_val, mask_y_val)]
        )
        return lgbmInstance, study.best_value
    
    def analyze_perFilter(self):
        if not self.evaluateTestResults:
            raise ValueError("evaluateTestResults is set to False. Cannot analyze per filter.")
        
        self.establishFilterDict()
        
        print("Overall Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_train)
        print("Overall Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_val)
        print("Overall Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.y_test)
        
        for filterTuple in self.filterColumnsTuple_names_leq:
            filterName = filterTuple[0]
            is_leq = filterTuple[1]
            
            print("---------------------------------------------------------")
            print(f"Running filter: {filterName}   LEQ: {is_leq}")
            
            mask_train, mask_val, mask_test = self.establishMask(filterName, is_leq)
            masked_X_train = self.X_train[mask_train]
            masked_y_train = self.y_train[mask_train]
            masked_X_val = self.X_val[mask_val]
            masked_y_val = self.y_val[mask_val]
            masked_X_test = self.X_test[mask_test]
            masked_y_test = self.y_test[mask_test]
            
            if (
                len(np.unique(masked_y_test)) < 2
                or len(np.unique(masked_y_val)) < 2
                or len(np.unique(masked_y_train)) < 2
            ):
                print("Skipping filter due to insufficient classes.")
                continue
            
            print("Training Masked Label Distribution:")
            ModelAnalyzer().print_label_distribution(masked_y_train)
            print("Validation Masked Label Distribution:")
            ModelAnalyzer().print_label_distribution(masked_y_val)
            print("Testing Masked Label Distribution:")
            ModelAnalyzer().print_label_distribution(masked_y_test)
            
            lgbmPostOptuna, _ = self.__run_OptunaOnFiltered(masked_X_train, masked_y_train, masked_X_val, masked_y_val)
            
            ModelAnalyzer().print_feature_importance_LGBM(lgbmPostOptuna, self.featureColumnNames, 5)
            
            masked_y_pred_val = lgbmPostOptuna.predict(masked_X_val)
            masked_y_pred_proba_val = lgbmPostOptuna.predict_proba(masked_X_val)
            
            masked_y_pred_test = lgbmPostOptuna.predict(masked_X_test)
            masked_y_pred_proba_test = lgbmPostOptuna.predict_proba(masked_X_test)

            print("Validation Masked Classification Metrics:")
            ModelAnalyzer().print_classification_metrics(masked_y_val, masked_y_pred_val, masked_y_pred_proba_val)
            print("Testing Masked Classification Metrics:")
            ModelAnalyzer().print_classification_metrics(masked_y_test, masked_y_pred_test, masked_y_pred_proba_test)

    def predict(self, lgbModelsList: List[lgb.LGBMClassifier] = None):
        if not self.dataIsPrepared:
            raise ValueError("Data is not prepared. Please run prepareData() first.")
        
        self.establishFilterDict()
        
        if lgbModelsList is None:
            lgbModelsList = []
            best_values = []
            for filterTuple in self.filterColumnsTuple_names_leq:
                filterName = filterTuple[0]
                is_leq = filterTuple[1]
                
                mask_train, mask_val, mask_test = self.establishMask(filterName, is_leq)
                masked_X_train = self.X_train[mask_train]
                masked_y_train = self.y_train[mask_train]
                masked_X_val = self.X_val[mask_val]
                masked_y_val = self.y_val[mask_val]
                
                if (
                    len(np.unique(masked_y_val)) < 2 or
                    len(np.unique(masked_y_train)) < 2
                ):
                    print("Skipping filter due to insufficient classes.")
                    lgbmPostOptuna = None
                    lgbModelsList.append(lgbmPostOptuna)
                    continue
                
                lgbmPostOptuna, best_value = self.__run_OptunaOnFiltered(masked_X_train, masked_y_train, masked_X_val, masked_y_val)
                lgbModelsList.append(lgbmPostOptuna)
                best_values.append(best_value)
                
                print(f"  Filter: {filterName}, Best Value: {best_value}")
        
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
            
            for i, val in enumerate(self.filterColumnsTuple_names_leq):
                filterName = val[0]
                is_leq = val[1]
                columnToSubset = self.filterColumnsDict_namesToIndex[filterName]
                mask_quantile = np.quantile(self.X_test[:, columnToSubset], self.dataRatio)
                
                if is_leq and features[columnToSubset] < mask_quantile:
                    lgbmModel = lgbModelsList[i]
                    prediction = lgbmModel.predict([features])[0]
                    proba = lgbmModel.predict_proba([features])[0]
                    
                    print(f"Asset {ticker}: Prediction: {prediction}, Probabilities: {proba}")
                    
                elif not is_leq and features[columnToSubset] >= mask_quantile:
                    lgbmModel = lgbModelsList[i]
                    prediction = lgbmModel.predict([features])[0]
                    proba = lgbmModel.predict_proba([features])[0]
                    
                    print(f"Asset {ticker}: Prediction: {prediction}, Probabilities: {proba}")
    
                else:
                    print(f"Asset {ticker} does not pass filter {filterName}.")
            
            
            
        
                
        
                
            
                
                
        
        
        
        
        

"""
    def establishShallowTree_deprecated(self):
        if not self.dataIsPrepared:
            self.prepareData()
            formatted_date = datetime.datetime.now().strftime('%d%m%y')
            self.save_data('src/predictionModule/bin/', f'SubsetML_data_{formatted_date}')
            
        min_length=len(self.X_train)//20
        max_features_to_filter=2
        
        mask = np.ones(len(self.X_train), dtype=bool)
        chosen_filters = []
        
        indexYear = self.featureColumnNames.index("Seasonal_year")
        lenallYears = len(set((self.X_train[:, indexYear])))

        for _ in range(max_features_to_filter):
            best_combo = (None, None, None, -1) # (feat, sep, leq, best_proportion)
            X_masked = self.X_train[mask]
            y_masked = self.y_train[mask]

            for i, feat in enumerate(self.featureColumnNames):
                if i % 100 == 0:
                    print(f"Progress: {i/len(self.featureColumnNames)*100:.2f}%")
                # Pick fewer candidates (e.g., quantiles only) to reduce loops
                feat_vals = X_masked[:, i]
                if len(np.unique(feat_vals)) < 6:
                    candidates = np.unique(feat_vals)
                else:
                    qs = np.linspace(0.1, 0.9, 5)
                    candidates = np.quantile(feat_vals, qs)

                for sep in candidates:
                    for leq in [True, False]:
                        new_mask = mask & (self.X_train[:, i] <= sep if leq else self.X_train[:, i] >= sep)
                        if new_mask.sum() >= min_length and len(set(self.X_train[new_mask, indexYear])) == lenallYears:
                            prop_neg = np.mean(self.y_train[new_mask] == 0)
                            prop_pos = np.mean(self.y_train[new_mask] == 2)
                            best_prop = prop_pos
                            if best_prop > best_combo[3]:
                                best_combo = (feat, sep, leq, best_prop)

            if best_combo[0] is None:
                break
            
            chosen_filters.append(best_combo[:3])
            # Update global mask with the chosen filter
            feat, thr, op = best_combo[:3]
            col_idx = self.featureColumnNames.index(feat)
            mask &= (self.X_train[:, col_idx] <= thr) if op else (self.X_train[:, col_idx] >= thr)

        final_neg_prop = np.mean(self.y_train[mask] == 0)
        final_pos_prop = np.mean(self.y_train[mask] == 2)
        print("Chosen filters:", chosen_filters)
        print(f"Final size: {mask.sum()} | Neg Prop: {final_neg_prop:.4f} | Pos Prop: {final_pos_prop:.4f}")
"""