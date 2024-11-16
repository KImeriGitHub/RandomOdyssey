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
    def fourierML(assetspl: Dict[str, AssetDataPolars]):
        startTrainDate=pd.Timestamp(year=2009, month=1, day=4)
        endTrainDate=pd.Timestamp(year=2019, month=2, day=4)
        startTestDate=pd.Timestamp(year=2019, month=4, day=5)
        fourierTestML = FourierML(assetspl, 
                 trainStartDate = startTrainDate,
                 trainEndDate = endTrainDate,
                 testStartDate = startTestDate,
                 testEndDate= startTestDate+ pd.Timedelta(days=10))

        fourierTestML.prepareData()

        fourierTestML.save_data('src/predictionModule/bin', "fourier_09to19_halfSpare")

        fourierTestML.traintestXGBModel()
        print(fourierTestML.metadata)

        fourierTestML.save_data('src/predictionModule/bin', "fourier_09to19_halfSpare_xgb")

