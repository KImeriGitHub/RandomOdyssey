from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from strategy.StratLinearAscendRanked import StratLinearAscendRanked
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.Portfolio import Portfolio
from src.common.AssetDataPolars import AssetDataPolars
from src.predictionModule.CurveML import CurveML

import pandas as pd
from typing import Dict

class CollectionModels():
    def __init__():
        pass

    @staticmethod
    def curveML_swiss_10to20(assets: Dict[str, AssetDataPolars]):

        startDate=pd.Timestamp(year=2010,month=1,day=4)
        endDate=pd.Timestamp(year=2020,month=1,day=4)

        curveML = CurveML(assets, startDate, endDate)

        #curveML.traintestModel()

        #curveML.saveModel("src/predictionModule/bin", "curveML_swiss_10to20")

    @staticmethod
    def curveML_snp500_10to20(assets: Dict[str, AssetDataPolars]):
        startDate=pd.Timestamp(year=2010,month=1,day=4)
        endDate=pd.Timestamp(year=2020,month=1,day=4)

        curveML = CurveML(assets, startDate, endDate, numOfMonths=24)

        curveML.traintestCNNModel()

        #curveML.saveModel("src/predictionModule/bin", "curveML_snp500_10to20")

