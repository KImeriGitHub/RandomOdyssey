from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from src.strategy.StratLinearAscendRanked import StratLinearAscendRanked
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
        startTrainDate=pd.Timestamp(year=2011, month=5, day=1)
        endTrainDate=pd.Timestamp(year=2015, month=6, day=21)
        startTestDate=pd.Timestamp(year=2015, month=6, day=22)
        endTestDate=pd.Timestamp(year=2016, month=6, day=22)
        startValDate=pd.Timestamp(year=2016, month=6, day=23)
        endValDate=pd.Timestamp(year=2017, month=6, day=22)
        fourierML = FourierML(assetspl, 
                 trainStartDate = startTrainDate,
                 trainEndDate = endTrainDate,
                 testStartDate = startTestDate,
                 testEndDate = endTestDate,
                 valStartDate = startValDate,
                 valEndDate = endValDate)

        fourierML.prepareData()

        fourierML.save_data('src/predictionModule/bin', "fourier_tr11to15_te15to16_va16to17_twenthiesSpare_1000Coeff_13mon")
        print(fourierML.metadata)

    @staticmethod
    def fourierML_loadupData_xgb(assetspl: Dict[str, AssetDataPolars]):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', "fourier_tr11to15_te15to16_va16to17_twenthiesSpare_1000Coeff_13mon")

        xgb_params = {
            'n_estimators': 5000,
            'learning_rate': 0.01,
            "colsample_bytree": 0.1,
            "colsample_bylevel": 0.9,
            "max_depth": 7,
            "subsample": 0.9
        }

        fourierML.traintestXGBModel(xgb_params)
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', "fourier_tr11to15_te15to16_va16to17_twenthiesSpare_1000Coeff_13mon_xgb")

    @staticmethod
    def fourierML_loadupData_rp(assetspl: Dict[str, AssetDataPolars]):
        fourierML = FourierML(assetspl)

        fourierML.load_data('src/predictionModule/bin', "fourier_tr11to15_te15to16_va16to17_twenthiesSpare_1000Coeff_13mon")

        rp_params = {
            'num_random_features': 10000,
            'regularization': 30,
            'max_iter': 10,
            'verbose': True,
            'random_state': None
        }

        fourierML.traintestRPModel(rp_params)
        print(fourierML.metadata)
        fourierML.save_data('src/predictionModule/bin', "fourier_tr11to15_te15to16_va16to17_twenthiesSpare_1000Coeff_13mon_rp")