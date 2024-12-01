from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from src.strategy.StratLinearAscendRanked import StratLinearAscendRanked
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.Portfolio import Portfolio
from src.common.AssetDataPolars import AssetDataPolars
from src.predictionModule.FourierML import FourierML
from src.predictionModule.NextDayML import NextDayML
from src.predictionModule.ModelAnalyzer import ModelAnalyzer

import pandas as pd
from typing import Dict

class CollectionModels():
    def __init__():
        pass

    @staticmethod
    def fourierML_saveData(assetspl: Dict[str, AssetDataPolars], save_name:str):
        params = {
            'idxLengthOneMonth': 21,
            'fouriercutoff': 100,
            'spareDatesRatio': 1.0,
            'multFactor': 8,
            'daysAfterPrediction': +1,
            'numOfMonths': 13,
            'classificationInterval': [0.0045], 
        }
        
        startTrainDate=pd.Timestamp(year=2015, month=1, day=4, tz='UTC')
        endTrainDate=pd.Timestamp(year=2015, month=2, day=4, tz='UTC')
        startTestDate=pd.Timestamp(year=2015, month=2, day=5, tz='UTC')
        endTestDate=pd.Timestamp(year=2015, month=2, day=11, tz="UTC")
        startValDate=pd.Timestamp(year=2015, month=2, day=12, tz="UTC")
        endValDate=pd.Timestamp(year=2015, month=2, day=19, tz="UTC")
        fourierML = FourierML(assetspl, 
                 trainStartDate = startTrainDate,
                 trainEndDate = endTrainDate,
                 testStartDate = startTestDate,
                 testEndDate = endTestDate,
                 valStartDate = startValDate,
                 valEndDate = endValDate,
                 params = params)

        fourierML.prepareData()

        fourierML.save_data('src/predictionModule/bin', save_name)
        print(fourierML.metadata)

    @staticmethod
    def fourierML_loadupData_xgb(assetspl: Dict[str, AssetDataPolars], loadup_name: str):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', loadup_name)

        xgb_params = {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.05
        }

        fourierML.traintestXGBModel(xgb_params, name_model_name=loadup_name+"_xgbModel", name_model_path="src/predictionModule/bin")
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', loadup_name)
        
        ModelAnalyzer(fourierML).plot_label_distribution()
        ModelAnalyzer(fourierML).plot_feature_importance()

    @staticmethod
    def fourierML_loadupData_rp(assetspl: Dict[str, AssetDataPolars], loadup_name: str):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', loadup_name)

        rp_params = {
            'num_random_features': 10000,
            'regularization': 30,
            'max_iter': 10,
            'verbose': True,
            'random_state': None
        }

        fourierML.traintestRPModel(rp_params, name_model_name=loadup_name+"_rpModel", name_model_path="src/predictionModule/bin")
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', loadup_name)
        
    @staticmethod
    def fourierML_loadupData_LSTM(assetspl: Dict[str, AssetDataPolars], loadup_name: str):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', loadup_name)
        
        lstm_params = {
            'units': 128,
            'dropout': 0.2,
            'dense_units': 64,
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'mean_absolute_error',
            'metrics': ['mae'],
            'epochs': 20,
            'batch_size': 128
        }

        fourierML.traintestLSTMModel(lstm_params, name_model_name=loadup_name+"_lstmModel", name_model_path="src/predictionModule/bin")
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', loadup_name)
        
        ModelAnalyzer(fourierML).plot_lstm_absolute_diff_histogram()
        
    @staticmethod
    def NextDayML_saveData(assetspl: Dict[str, AssetDataPolars], save_name:str):
        params = {
            'spareDatesRatio': 0.1,
            'daysAfterPrediction': 1,
            'monthsHorizon': 13,
            'timesteps': 5,
        }
        
        startTrainDate=pd.Timestamp(year=2011, month=9, day=4, tz='UTC')
        endTrainDate=pd.Timestamp(year=2015, month=2, day=4, tz='UTC')
        startTestDate=pd.Timestamp(year=2015, month=2, day=5, tz='UTC')
        endTestDate=pd.Timestamp(year=2016, month=2, day=11, tz="UTC")
        startValDate=pd.Timestamp(year=2016, month=2, day=12, tz="UTC")
        endValDate=pd.Timestamp(year=2016, month=2, day=19, tz="UTC")
        nextDayML = NextDayML(assetspl, 
                 trainStartDate = startTrainDate,
                 trainEndDate = endTrainDate,
                 testStartDate = startTestDate,
                 testEndDate = endTestDate,
                 valStartDate = startValDate,
                 valEndDate = endValDate,
                 params = params)

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
        
        ModelAnalyzer(nextDayML).plot_label_distribution()
        ModelAnalyzer(nextDayML).plot_feature_importance()
        
    @staticmethod
    def NextDayML_loadupData_lgbm(assetspl: Dict[str, AssetDataPolars], loadup_name: str):
        nextDayML = NextDayML(assetspl)

        nextDayML.load_data('src/predictionModule/bin', loadup_name)

        lgbm_params = {
                'n_estimators': 500,
                'num_leaves':32,
                'early_stopping_round':100,
                'learning_rate': 0.01,
                'max_depth': 5,
                'subsample': 1,
                'feature_fraction': 0.1
        }

        nextDayML.traintestLGBMModel(lgbm_params, name_model_name=loadup_name+"_lgbmModel", name_model_path="src/predictionModule/bin")
        print(nextDayML.metadata)
        nextDayML.save_data('src/predictionModule/bin', loadup_name)
        
        ModelAnalyzer(nextDayML).plot_label_distribution()
        ModelAnalyzer(nextDayML).plot_feature_importance()
        
    @staticmethod
    def NextDayML_loadupData_LSTM(assetspl: Dict[str, AssetDataPolars], loadup_name: str):
        nextDayML = NextDayML(assetspl)

        nextDayML.load_data('src/predictionModule/bin', loadup_name)
        
        lstm_params = {
            'units': 128,
            'dropout': 0.2,
            'dense_units': 64,
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'mean_absolute_error',
            'metrics': ['mae'],
            'epochs': 20,
            'batch_size': 128
        }

        nextDayML.traintestLSTMModel(lstm_params, name_model_name=loadup_name+"_lstmModel", name_model_path="src/predictionModule/bin")
        print(nextDayML.metadata)
        nextDayML.save_data('src/predictionModule/bin', loadup_name)
        
        ModelAnalyzer(nextDayML).plot_lstm_absolute_diff_histogram()