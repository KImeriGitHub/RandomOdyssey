from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from strategy.StratLinearAscendRanked import StratLinearAscendRanked
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.Portfolio import Portfolio
from src.common.AssetDataPolars import AssetDataPolars
from src.predictionModule.FourierML import FourierML

import pandas as pd
from typing import Dict

class CollectionModels():
    def __init__():
        pass

    @staticmethod
    def fourierML_saveData(assetspl: Dict[str, AssetDataPolars]):
        startTrainDate=pd.Timestamp(year=2015, month=5, day=1)
        endTrainDate=pd.Timestamp(year=2016, month=6, day=21)
        startTestDate=pd.Timestamp(year=2016, month=6, day=22)
        fourierML = FourierML(assetspl, 
                 trainStartDate = startTrainDate,
                 trainEndDate = endTrainDate,
                 testStartDate = startTestDate)

        fourierML.prepareData()

        fourierML.save_data('src/predictionModule/bin', "fourier_15to16_halfSpare_1000Coeff_19mon")
        print(fourierML.metadata)

    @staticmethod
    def fourierML_loadupData_xgb(assetspl: Dict[str, AssetDataPolars]):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', "fourier_15to16_halfSpare_1000Coeff_19mon")

        xgb_params = {
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.05
        }

        fourierML.traintestXGBModel(xgb_params)
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', "fourier_15to16_halfSpare_1000Coeff_19mon_xgb")

    @staticmethod
    def fourierML_loadupData_rp(assetspl: Dict[str, AssetDataPolars]):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', "fourier_15to16_halfSpare_1000Coeff_19mon")

        rp_params = {
            'num_random_features': 5000,
            'regularization': 30,
            'max_iter': 10,
            'verbose': True,
            'random_state': None
        }

        fourierML.traintestRPModel(rp_params)
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', "fourier_15to16_halfSpare_1000Coeff_19mon_rp")