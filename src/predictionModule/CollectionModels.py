from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from src.strategy.StratLinearAscendRanked import StratLinearAscendRanked
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.Portfolio import Portfolio
from src.common.AssetDataPolars import AssetDataPolars
from src.predictionModule.AkinDistriML import AkinDistriML
from src.predictionModule.SubsetML import SubsetML
from src.predictionModule.NextDayML import NextDayML
from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.predictionModule.ConditionalML import ConditionalML

import pandas as pd
import numpy as np
from typing import Dict
import optuna
import logging
from datetime import datetime

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

class CollectionModels():
    # Prepare Dates
    @staticmethod
    def sample_spare_dates(start_date, end_date, ratio=0.1, fallback_days=5):
        """Return a DatetimeIndex of randomly sampled spare dates."""
        if start_date is None:
            # No dates
            return pd.DatetimeIndex([])
        if end_date is None:
            # Fallback range if end_date isn't provided
            date_range = pd.date_range(start_date, start_date + pd.Timedelta(days=fallback_days), freq='B')
            return pd.DatetimeIndex([date_range[0]])
        date_range = pd.date_range(start_date, end_date, freq='B')
        n_samples = max(int(len(date_range) * ratio), 1)
        return pd.DatetimeIndex(np.random.choice(date_range, n_samples, replace=False))
            
    @staticmethod
    def NextDayML_saveData(
            assetspl: Dict[str, AssetDataPolars], 
            save_name:str, 
            trainDates: pd.TimedeltaIndex=None, 
            valDates: pd.TimedeltaIndex=None, 
            testDates: pd.TimedeltaIndex=None,
            params = None):

        nextDayML = NextDayML(assetspl, 
                trainDates = trainDates,
                valDates = valDates,
                testDates = testDates,
                params = params,
                enableTimeSeries = False)

        nextDayML.prepareData()

        nextDayML.save_data('src/predictionModule/bin', save_name)
        print(nextDayML.metadata)
        
        
    @staticmethod
    def NextDayML_loadupData_xgb(assetspl: Dict[str, AssetDataPolars], loadup_name: str):
        nextDayML = NextDayML(assetspl)

        nextDayML.load_data('src/predictionModule/bin', loadup_name)

        xgb_params = {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 5,
                'subsample': 1,
                'colsample_bytree': 0.1
        }

        nextDayML.traintestXGBModel(xgb_params, name_model_name=loadup_name+"_xgbModel", name_model_path="src/predictionModule/bin")
        print(nextDayML.metadata)
        nextDayML.save_data('src/predictionModule/bin', loadup_name)
        
    @staticmethod
    def NextDayML_loadupData_lgbm(assetspl: Dict[str, AssetDataPolars], loadup_name: str, params = None):
        nextDayML = NextDayML(assetspl)
        nextDayML.load_data('src/predictionModule/bin', loadup_name)

        ModelAnalyzer().print_label_distribution(nextDayML.y_val, nextDayML.y_test)

        def objective(trial):
            # 2. Suggest values of the hyperparameters using a trial object.
            lgbm_params = {
                'verbosity': -1,
                #'n_jobs': -1,
                'boosting_type': 'gbdt',
                'early_stopping_rounds': 100,
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'lambda_l1': 0.9,
                'lambda_l2': 0.9,
                'num_leaves': trial.suggest_int('num_leaves', 8, 512),
                'max_depth': 15,
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.9, log=True),
            }

            nextDayML.traintestLGBMModel(lgbm_params)
            return nextDayML.metadata['LGBMModel_accuracy_val']

        # 3. Create a study object and optimize the objective function.
        study = optuna.create_study(direction='maximize')
        n_trials = params['optuna_trials'] if params is not None else 10
        study.optimize(objective, n_trials = n_trials)
        
        best_trial = study.best_trial
        print("Best Trial:")
        print(f"  Value (Accuracy): {best_trial.value}")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        lgbm_params = {
            'verbosity': -1,
            #'n_jobs': -1,
            'boosting_type': 'gbdt',
            'early_stopping_rounds': 100,
            'n_estimators': best_trial.params['n_estimators'],
            'lambda_l1': 0.9,
            'lambda_l2': 0.9,
            'num_leaves': best_trial.params['num_leaves'],
            'max_depth': 15,
            'learning_rate': best_trial.params['learning_rate'],
        }
        
        nextDayML.traintestLGBMModel(lgbm_params, name_model_name=loadup_name+"_lgbmModel", name_model_path="src/predictionModule/bin")
        print(nextDayML.metadata)
        nextDayML.save_data('src/predictionModule/bin', loadup_name)
        
        y_pred = nextDayML.LGBMModel.predict(nextDayML.X_test)
        y_pred_proba = nextDayML.LGBMModel.predict_proba(nextDayML.X_test)
        ModelAnalyzer().print_feature_importance_LGBM(nextDayML, 100)
        ModelAnalyzer().print_classification_metrics(nextDayML.y_test, y_pred, y_pred_proba)
        
    @staticmethod
    def NextDayML_loadupData_lgbm_noOptuna(assetspl: Dict[str, AssetDataPolars], loadup_name: str, params = None):
        nextDayML = NextDayML(assetspl)
        nextDayML.load_data('src/predictionModule/bin', loadup_name)

        ModelAnalyzer().print_label_distribution(nextDayML.y_val, nextDayML.y_test)

        lgbm_params = {
            'verbosity': -1,
            'n_jobs': -1,
            'boosting_type': 'gbdt',
            'early_stopping_rounds': 100,
            'n_estimators': 500,
            'lambda_l1': 0.9,
            'lambda_l2': 0.9,
            'num_leaves': 400,
            'max_depth': 15,
            'learning_rate': 0.99,
        }
        
        nextDayML.traintestLGBMModel(lgbm_params, name_model_name=loadup_name+"_lgbmModel", name_model_path="src/predictionModule/bin")
        print(nextDayML.metadata)
        nextDayML.save_data('src/predictionModule/bin', loadup_name)
        
        y_pred = nextDayML.LGBMModel.predict(nextDayML.X_test)
        y_pred_proba = nextDayML.LGBMModel.predict_proba(nextDayML.X_test)
        ModelAnalyzer().print_feature_importance_LGBM(nextDayML, 100)
        ModelAnalyzer().print_classification_metrics(nextDayML.y_test, y_pred, y_pred_proba)

    @staticmethod
    def NextDayML_loadupData_LSTM(assetspl: Dict[str, AssetDataPolars], loadup_name: str):
        nextDayML = NextDayML(assetspl)

        nextDayML.load_data('src/predictionModule/bin', loadup_name)
        
        lstm_params = {
            'units': 512,
            'dropout': 0.1,
            'dense_units': 64,
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'mean_absolute_error',
            'metrics': ['mae'],
            'epochs': 1000,
            'batch_size': 256,
            'early_stopping_round':150,
        }

        nextDayML.traintestLSTMModel(lstm_params, name_model_name=loadup_name+"_lstmModel", name_model_path="src/predictionModule/bin")
        print(nextDayML.metadata)
        nextDayML.save_data('src/predictionModule/bin', loadup_name)
        
        ModelAnalyzer(nextDayML).plot_lstm_absolute_diff_histogram()
        
    @staticmethod
    def ConditionalML_saveData(
            assetspl: Dict[str, AssetDataPolars], 
            save_name:str, 
            trainDates: pd.TimedeltaIndex=None, 
            valDates: pd.TimedeltaIndex=None, 
            testDates: pd.TimedeltaIndex=None,
            params = None):

        conditionalML = ConditionalML(assetspl, 
                trainDates = trainDates,
                valDates = valDates,
                testDates = testDates,
                params = params,
                enableTimeSeries = False)

        conditionalML.prepareData()

        conditionalML.save_data('src/predictionModule/bin', save_name)
        print(conditionalML.metadata)
        
    @staticmethod
    def CondtionalML_loadupData_lgbm(assetspl: Dict[str, AssetDataPolars], loadup_name: str, params = None):
        conditionalML = ConditionalML(assetspl)
        conditionalML.load_data('src/predictionModule/bin', loadup_name)

        print("Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(conditionalML.y_train)
        
        print("Validation Label Distribution:")
        ModelAnalyzer().print_label_distribution(conditionalML.y_val)

        print("Testing Label Distribution:")
        ModelAnalyzer().print_label_distribution(conditionalML.y_test)

        def objective(trial):
            # 2. Suggest values of the hyperparameters using a trial object.
            lgbm_params = {
                'verbosity': -1,
                'n_jobs': -1,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'early_stopping_rounds': 10,
                'n_estimators': 100,
                'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 0.9, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 0.9, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.001, 0.01),
                #'bagging_fraction': trial.suggest_float('bagging_fraction', 0.001, 0.9, log=True),
                #'bagging_freq': trial.suggest_int('bagging_freq', 0, 30),
                'num_leaves': trial.suggest_int('num_leaves', 512, 2048),
                'max_depth': 15,
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            }
            conditionalML.traintestLGBMModel(lgbm_params)
            y_val_pred = conditionalML.LGBMModel.predict(conditionalML.X_val)
            cm:np.array = confusion_matrix(conditionalML.y_val, y_val_pred, labels=np.unique(conditionalML.y_val))
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
            return conditionalML.metadata['LGBMModel_accuracy_val']

        # 3. Create a study object and optimize the objective function.
        study = optuna.create_study(direction='maximize')
        n_trials = params['optuna_trials'] if params['optuna_trials'] is not None else 10
        study.optimize(objective, n_trials = n_trials, timeout=60*60*2)
        
        best_trial = study.best_trial
        print("Best Trial:")
        print(f"  Value (Accuracy): {best_trial.value}")
        print("  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        lgbm_params = {
            'verbosity': -1,
            'n_jobs': -1,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'early_stopping_rounds': 10,
            'n_estimators': 100,
            'lambda_l1':best_trial.params['lambda_l1'],
            'lambda_l2': best_trial.params['lambda_l2'],
            'feature_fraction': best_trial.params['feature_fraction'],
            #'bagging_fraction': best_trial.params['bagging_fraction'],
            #'bagging_freq': best_trial.params['bagging_freq'],
            'num_leaves': best_trial.params['num_leaves'],
            'max_depth': 15,
            'learning_rate': best_trial.params['learning_rate'],
        }
        
        conditionalML.traintestLGBMModel(lgbm_params, name_model_name=loadup_name+"_lgbmModel", name_model_path="src/predictionModule/bin")
        print(conditionalML.metadata)
        conditionalML.save_data('src/predictionModule/bin', loadup_name)
        
        y_pred_val = conditionalML.LGBMModel.predict(conditionalML.X_val)
        y_pred_val_proba = conditionalML.LGBMModel.predict_proba(conditionalML.X_val)
        y_pred_test = conditionalML.LGBMModel.predict(conditionalML.X_test)
        y_pred_test_proba = conditionalML.LGBMModel.predict_proba(conditionalML.X_test)
        
        print("Feature Importance:")
        ModelAnalyzer().print_feature_importance_LGBM(conditionalML, 100)
        print()
        print("Validation Metrics:")
        ModelAnalyzer().print_classification_metrics(conditionalML.y_val, y_pred_val, y_pred_val_proba)
        print("Test Metrics:")
        ModelAnalyzer().print_classification_metrics(conditionalML.y_test, y_pred_test, y_pred_test_proba)
        
    @staticmethod
    def SubsetML_saveData(
            assetspl: Dict[str, AssetDataPolars], 
            save_name: str, 
            test_date: pd.Timestamp,
            spareRatio: float = 0.5,
            params = None):

        subsetML = SubsetML(assetspl, 
                spareRatio=spareRatio,
                params = params,
                test_start_date=test_date)

        subsetML.prepareData()

        subsetML.save_data('src/predictionModule/bin', save_name)
        print(subsetML.metadata)
        
    @staticmethod
    def SubsetML_loadup_analyze(assetspl: Dict[str, AssetDataPolars], loadup_name: str, test_date: pd.Timestamp, params = None):
        subsetML = SubsetML(assetspl,test_start_date=test_date, params = params)
        subsetML.load_data('src/predictionModule/bin', loadup_name)
        
        subsetML.analyze_perFilter()
        
    @staticmethod
    def SubsetML_loadup_predict(
            assetspl: Dict[str, AssetDataPolars], 
            loadup_name: str, 
            test_date: pd.Timestamp,
            params = None):
        
        subsetML = SubsetML(assetspl,test_start_date=test_date, params=params)
        subsetML.load_data('src/predictionModule/bin', loadup_name)
        
        subsetML.predict()
        
    @staticmethod
    def AkinDistriML_saveData(
            assetspl: Dict[str, AssetDataPolars], 
            save_name: str, 
            test_date: pd.Timestamp,
            evaluateTestResults = True,
            params = None, 
            logger: logging.Logger = None):

        akinML = AkinDistriML(assetspl,
                params = params,
                test_date=test_date,
                gatherTestResults = evaluateTestResults, 
                logger = logger)

        startTime = datetime.now()
        akinML.prepareData()
        endTime = datetime.now()
        logger.info(f"Data Preparation completed in: {endTime - startTime}")

        akinML.save_data('src/predictionModule/bin', save_name)
        print(akinML.metadata)
        
    @staticmethod
    def AkinDistriML_loadup_analyze(assetspl: Dict[str, AssetDataPolars], loadup_name: str, test_date: pd.Timestamp, params = None, logger = None):
        akinML = AkinDistriML(assetspl,test_date=test_date, params = params, gatherTestResults=True, logger = logger)
        akinML.load_data('src/predictionModule/bin', loadup_name)
        
        return akinML.analyze()
    
    @staticmethod
    def AkinDistriML_saveData_predict(
        assetspl: Dict[str, AssetDataPolars], 
        save_name: str, 
        test_date: pd.Timestamp, 
        params = None, 
        logger:logging.Logger = None
    ):
        akinML = AkinDistriML(
            assetspl,
            params = params,
            test_date=test_date,
            gatherTestResults = False,
            logger = logger
        )

        startTime = datetime.now()
        akinML.prepareData()
        endTime = datetime.now()
        logger.info(f"Data Preparation completed in: {endTime - startTime}")

        akinML.save_data('src/predictionModule/bin', save_name)
        print(akinML.metadata)
        
        return akinML.predict()
    
    @staticmethod
    def AkinDistriML_loadUpData_predict(assetspl: Dict[str, AssetDataPolars], loadup_name: str, test_date: pd.Timestamp, params = None, logger = None):
        akinML = AkinDistriML(
            assetspl,
            params = params,
            test_date=test_date,
            gatherTestResults = False,
            logger = logger
        )

        akinML.load_data('src/predictionModule/bin', loadup_name)
        
        print(akinML.metadata)
        
        return akinML.predict()