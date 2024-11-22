from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from src.strategy.StratLinearAscendRanked import StratLinearAscendRanked
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.Portfolio import Portfolio
from src.common.AssetDataPolars import AssetDataPolars
from src.predictionModule.FourierML import FourierML
from src.predictionModule.ModelAnalyzer import ModelAnalyzer

import pandas as pd
from typing import Dict

class CollectionModels():
    def __init__():
        pass

    @staticmethod
    def fourierML_saveData(assetspl: Dict[str, AssetDataPolars]):
        params = {
            'idxLengthOneMonth': 21,
            'fouriercutoff': 100,
            'spareDatesRatio': 1.0,
            'multFactor': 8,
            'lenClassInterval': 1,
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

        fourierML.save_data('src/predictionModule/bin', "fourier_twomonth_test2015")
        print(fourierML.metadata)

    @staticmethod
    def fourierML_loadupData_xgb(assetspl: Dict[str, AssetDataPolars]):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', "fourier_twomonth_test2015")

        xgb_params = {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.05
        }

        fourierML.traintestXGBModel(xgb_params, name_model_name="fourier_twomonth_test2015_xgbModel", name_model_path="src/predictionModule/bin")
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', "fourier_twomonth_test2015")
        
        ModelAnalyzer(fourierML).plot_label_distribution()
        ModelAnalyzer(fourierML).plot_feature_importance()

    @staticmethod
    def fourierML_loadupData_rp(assetspl: Dict[str, AssetDataPolars]):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', "fourier_twomonth_test2015")

        rp_params = {
            'num_random_features': 10000,
            'regularization': 30,
            'max_iter': 10,
            'verbose': True,
            'random_state': None
        }

        fourierML.traintestRPModel(rp_params, name_model_name="fourier_twomonth_test2015_xgbModel", name_model_path="src/predictionModule/bin")
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', "fourier_twomonth_test2015")