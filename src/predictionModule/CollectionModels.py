from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from strategy.StratLinearAscendRanked import StratLinearAscendRanked
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.Portfolio import Portfolio
from src.common.AssetDataPolars import AssetDataPolars
from predictionModule.FourierML import FourierML

import pandas as pd
from typing import Dict

class CollectionModels():
    def __init__():
        pass

    @staticmethod
    def fourierML_snp500_10to20(assets: Dict[str, AssetDataPolars]):
        startDate=pd.Timestamp(year=2019, month=9, day=4)
        endDate=pd.Timestamp(year=2020, month=1, day=4)

        fourierML = FourierML(assets, startDate, endDate)

        fourierML.traintestRPModel()

        #fourierML.saveCNNModel("src/predictionModule/bin", "fourierML_snp500_10to20")

