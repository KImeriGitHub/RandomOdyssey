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
        'idxAfterPrediction': 21,
        'monthsHorizon': 13,
        'timesteps': 5,
        'classificationInterval': [0.05], 
    }

    def __init__(self, assets: Dict[str, AssetDataPolars], 
                 test_start_date: pd.Timestamp,
                 spareRatio: float = 0.5,
                 params: dict = None,
                 evaluateTestResults: bool = True):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets
        
        self.lagList = [1,2,3,5,7,10,13,16,21,45,60,102,200,255,290]
        self.featureColumnNames = []
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.evaluateTestResults = evaluateTestResults
        self.test_start_date = test_start_date
        
        trainingInterval_days = 120
        testInterval_days = 5
        val_idxDays = 20
        self.quantil: float = 0.2
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
        self.trainDates, self.valDates = self.__calculate_dates(train_start_date, train_end_date, val_idxDays)
        
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
        
    def __calculate_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp, val_ratio: float):
        # Create a continuous date range
        date_range = pd.date_range(start_date, end_date, freq='B')

        # Time-based split (e.g. last 20% for validation)
        n_train = int(len(date_range) * (1 - val_ratio))
        train_dates = date_range[:n_train]
        val_dates = date_range[n_train:]
        return train_dates, val_dates
    
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
                if len(asset.shareprice["Date"]) <= aidx + self.idxAfterPrediction:
                    print(f"Asset {ticker} does not have enough data to calculate target on date {date}.")
                    continue

                features = self.getFeatures(asset, featureMain, date, aidx)
                target = self.getTarget(asset, featureMain, date, aidx)

                Xtrain.append(features)
                ytrain.append(target)
                
            # Prepare Val Data
            for date in self.valDates:
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
                target = self.getTarget(asset, featureMain, date, aidx)

                Xval.append(features)
                yval.append(target)

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
                    target = self.getTarget(asset, featureMain, date, aidx)
                    ytest.append(target)
                else:
                    ytest.append(-1)

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

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test)
        self.X_val, self.y_val = shuffle(self.X_val, self.y_val)
        
        self.dataIsPrepared = True

    def establishFilterDict(self):
        q = self.quantil
        
        self.filterColumnsTuple_names_quantilDown_quantilUp = [
            ("Fourier_Price_RSME", 0, q),
            #("FinData_quar_surprise", q, 1),
            ("MathFeature_Drawdown", 1-q, 1),
            #("FinData_quar_reportedEPS_lag_qm1", q, 1),
            #("FinData_quar_surprise_lagquot_qm1", False),
            #("FinData_quar_grossProfit", False),
            #("FinData_quar_grossProfit_lag_qm1", False),
            #("FinData_metrics_log_forward_pe_ratio", True),
            #("FinData_metrics_log_forward_pe_ratio_lagquot_m1", 0, q),
            #("FinData_metrics_log_trailing_pe_ratio_lag_m1", True),
            #("FinData_metrics_log_trailing_pe_ratio", 0, q),
            #("FinData_ann_grossProfit", q, 1),
            #("FinData_ann_grossProfit_lag_qm1", False),
            #("FinData_ann_reportedEPS", True),
            #("Fourier_Price_AbsCoeff_1", False),
            #("Fourier_Price_AbsCoeff_2", True),
            ("FeatureTA_trend_macd", 1-q, 1),
            #("FeatureTA_trend_macd_signal", True),
            #("FeatureTA_Open", q, 1),
        ]
        
        self.filterColumnsDict_namesToIndex = {}
        for val in self.filterColumnsTuple_names_quantilDown_quantilUp:
            self.filterColumnsDict_namesToIndex[val[0]] = self.featureColumnNames.index(val[0])
            
    def establishMask(self, filterName: str, lowerLevel: float, upperLevel: float):
        columnToSubset = self.filterColumnsDict_namesToIndex[filterName]
        
        mask_quantilUp = np.quantile(self.X_test[:, columnToSubset], upperLevel)
        mask_quantilDown = np.quantile(self.X_test[:, columnToSubset], lowerLevel)
        
        if lowerLevel < 1e-10:
            mask_train = self.X_train[:, columnToSubset] <= mask_quantilUp
            mask_val = self.X_val[:, columnToSubset] <= mask_quantilUp
            mask_test = self.X_test[:, columnToSubset] <= mask_quantilUp
            
            return mask_train, mask_val, mask_test
            
        if upperLevel > 1-1e-10:
            mask_train = self.X_train[:, columnToSubset] >= mask_quantilDown
            mask_val = self.X_val[:, columnToSubset] >= mask_quantilDown
            mask_test = self.X_test[:, columnToSubset] >= mask_quantilDown
            
            return mask_train, mask_val, mask_test
        
        mask_train = (self.X_train[:, columnToSubset] <= mask_quantilUp) & (self.X_train[:, columnToSubset] >= mask_quantilDown)
        mask_val = (self.X_val[:, columnToSubset] <= mask_quantilUp) & (self.X_val[:, columnToSubset] >= mask_quantilDown)
        mask_test = (self.X_test[:, columnToSubset] <= mask_quantilUp) & (self.X_test[:, columnToSubset] >= mask_quantilDown)
        
        return mask_train, mask_val, mask_test
    
    def __run_OptunaOnFiltered(self, mask_X_Train, mask_y_Train, mask_X_val, mask_y_val, enablePrint: bool = False):
        def objective(trial: optuna.Trial):
            lgbm_params = {
                'verbosity': -1,
                'n_jobs': -1,
                'is_unbalance': True,
                'metric': 'binary_logloss',
                'lambda_l1': 1.0,
                'lambda_l2': 1.0,
                'n_estimators': 1000,
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 30, 100),
                'feature_fraction_bynode': trial.suggest_float('feature_fraction_bynode', 0.001, 0.01),
                'feature_fraction': trial.suggest_categorical('feature_fraction', [0.05,0.1,0.15]),
                'num_leaves': trial.suggest_int('num_leaves', 32, 256),
                'max_depth': trial.suggest_int('max_depth', 6, 16),
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
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
            return np.sum(per_class_accuracy)

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
            'metric': 'binary_logloss',
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'n_estimators': 1000,
            'min_data_in_leaf': study.best_trial.params['min_data_in_leaf'],
            'feature_fraction_bynode': study.best_trial.params['feature_fraction_bynode'],
            'feature_fraction': study.best_trial.params['feature_fraction'],
            'num_leaves': study.best_trial.params['num_leaves'],
            'max_depth': study.best_trial.params['max_depth'],
            'learning_rate': study.best_trial.params['learning_rate'],
            'random_state': 41,
        }
        # Final training on combined data, weighting validation set
        X_final = np.concatenate([mask_X_Train, mask_X_val])
        y_final = np.concatenate([mask_y_Train, mask_y_val])
        w_train = np.ones(len(mask_y_Train))
        w_val = np.ones(len(mask_y_val)) * self.params['optuna_weight']
        w_final = np.concatenate([w_train, w_val])

        lgbmInstance = lgb.LGBMClassifier(**best_params)
        lgbmInstance.fit(X_final, y_final, sample_weight=w_final)        
        return lgbmInstance, study.best_value
    
    def __establish_lgbmList(self):
        lgbModelsList = []
        best_values = []
        for filterTuple in self.filterColumnsTuple_names_quantilDown_quantilUp:
            filterName = filterTuple[0]
            
            mask_train, mask_val, _ = self.establishMask(filterName, filterTuple[1], filterTuple[2])
            masked_X_train = self.X_train[mask_train]
            masked_y_train = self.y_train[mask_train]
            masked_X_val = self.X_val[mask_val]
            masked_y_val = self.y_val[mask_val]
            
            if (
                len(np.unique(masked_y_val)) < 2 or
                len(np.unique(masked_y_train)) < 2
            ):
                print("__establish_lgbmList: Skipping filter due to insufficient classes.")
                lgbmPostOptuna = None
                lgbModelsList.append(lgbmPostOptuna)
                continue
            
            lgbmPostOptuna, best_value = self.__run_OptunaOnFiltered(masked_X_train, masked_y_train, masked_X_val, masked_y_val)
            lgbModelsList.append(lgbmPostOptuna)
            best_values.append(best_value)
            
            #print(f"  Filter: {filterName}, Best Value: {best_value}")
            
        self.lgbModelsList = lgbModelsList
        self.best_values = best_values
    
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
        
        for filterTuple in self.filterColumnsTuple_names_quantilDown_quantilUp:
            filterName = filterTuple[0]
            
            print("---------------------------------------------------------")
            print(f"Running filter: {filterName}")
            
            mask_train, mask_val, mask_test = self.establishMask(filterName, filterTuple[1], filterTuple[2])
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
            
            lgbmPostOptuna, _ = self.__run_OptunaOnFiltered(masked_X_train, masked_y_train, masked_X_val, masked_y_val, enablePrint=True)
            
            ModelAnalyzer().print_feature_importance_LGBM(lgbmPostOptuna, self.featureColumnNames, 5)
            
            masked_y_pred_val = lgbmPostOptuna.predict(masked_X_val)
            masked_y_pred_proba_val = lgbmPostOptuna.predict_proba(masked_X_val)
            
            masked_y_pred_test = lgbmPostOptuna.predict(masked_X_test)
            masked_y_pred_proba_test = lgbmPostOptuna.predict_proba(masked_X_test)
            
            print("Predicted Validation Label Distribution:")
            ModelAnalyzer().print_label_distribution(masked_y_pred_val)
            print("Predicted Testing Label Distribution:")
            ModelAnalyzer().print_label_distribution(masked_y_pred_test)
            
            if (
                len(np.unique(masked_y_pred_val)) < 2
                or len(np.unique(masked_y_pred_test)) < 2
            ):
                print("Skipping printing due to insufficient predicted classes.")
                continue

            print("Validation Masked Classification Metrics:")
            ModelAnalyzer().print_classification_metrics(masked_y_pred_val, masked_y_val, masked_y_pred_proba_val)
            print("Testing Masked Classification Metrics:")
            ModelAnalyzer().print_classification_metrics(masked_y_pred_test, masked_y_test, masked_y_pred_proba_test)
            
            # Top 5 highest probabilities in the second column (>50%)
            second_col_probs_test = masked_y_pred_proba_test[:, 1]
            top_5_indices = np.argsort(second_col_probs_test)[-5:][::-1]
            selected_top5_indices = [i for i in top_5_indices if second_col_probs_test[i] > 0.5]
            selected_over80_indices = [i for i in range(len(second_col_probs_test)) if second_col_probs_test[i] > 0.8]

            if selected_top5_indices:
                selected_true_labels = masked_y_test[selected_top5_indices]
                accuracy_top_5_above_50 = np.mean(selected_true_labels == 1)
                print(f"Accuracy of top up to 5 (prob > 50%) in test set: {accuracy_top_5_above_50:.2%}")
            else:
                print("No test predictions above 50% probability for class 1 among the top 5 rows.")
                
            if selected_over80_indices:
                selected_true_labels = masked_y_test[selected_over80_indices]
                accuracy_above_80 = np.mean(selected_true_labels == 1)
                print(f"Accuracy of prob > 80% in test set: {accuracy_above_80:.2%}")
            else:
                print("No test predictions above 80% probability for class 1.")
        

    def predict_perFilter(self, lgbModelsList: List[lgb.LGBMClassifier] = None):
        if not self.dataIsPrepared:
            raise ValueError("Data is not prepared. Please run prepareData() first.")
        
        self.establishFilterDict()
        
        if self.lgbModelsList is None or self.best_values is None:
            self.__establish_lgbmList()
        
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
        
        if self.lgbModelsList is None or self.best_values is None:
            self.__establish_lgbmList()
        
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
                
        
        