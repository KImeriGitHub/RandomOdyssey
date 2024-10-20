from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from strategy.StratLinearAscendRanked import StratLinearAscendRanked
from strategy.StratCurvePrediction import StratCurvePrediction
from strategy.StratQuadraticAscendRanked import StratQuadraticAscendRanked
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.Portfolio import Portfolio
from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService

import pandas as pd
import polars as pl
from typing import Dict

class CollectionSimulations():
    def __init__():
        pass

    @staticmethod
    def BuyAndHold():
        # Load asset data
        assetG = AssetFileInOut("src/database").loadFromFile('GOOGL')
        assetA = AssetFileInOut("src/database").loadFromFile('AAPL')
        assetM = AssetFileInOut("src/database").loadFromFile('MSFT')

        # Define strategy
        #strategy = StratBuyAndHold(targetTickers=['AAPL', 'GOOGL', 'MSFT'])
        strategy = StratBuyAndHold(targetTickers=['AAPL'])

        # Set up simulation
        simulation = SimulatePortfolio(
            initialCash=10000,
            strategy=strategy,
            assets=[assetG, assetA, assetM],
            startDate=pd.Timestamp(2010,1,1),
            endDate=pd.Timestamp(2020,1,1),
        )

        # Run simulation
        simulation.run()

        # Analyze results
        analyzer = ResultAnalyzer(simulation.portfolio)
        analyzer.plot_portfolio_value()

    @staticmethod
    def LinearAscend():
        # Load asset data
        #tickers = ['GOOGL', 'AAPL', 'MSFT', 'IRM', 'T', 
        #            'KO', 'AMZN', 'NVO', 'NVDA', 'HRB', 
        #            "WARN.SW", "HBLN.SW", "GRKP.SW", "ABBN.SW", "GF.SW"]
        #tickers = ['GOOGL', 'AAPL', 'MSFT']
        assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_over20years")

        # Convert to Polars for speedup
        assetspl: Dict[str, AssetDataPolars] = {}
        for ticker, asset in assets.items():
            assetspl[ticker]= AssetDataService.to_polars(asset)

        # Define strategy
        initialCash=10000.0
        strategy = StratLinearAscendRanked(num_months = 1, num_choices= 1)

        # Set up simulation
        simulation = SimulatePortfolio(
            portfolio = Portfolio(cash = initialCash),
            strategy=strategy,
            assets=assetspl,
            startDate=pd.Timestamp(2005,1,4),
            endDate=pd.Timestamp(2024,10,4),
        )

        # Run simulation
        simulation.run()

        # Analyze results
        analyzer = ResultAnalyzer(simulation.portfolio)
        analyzer.plot_portfolio_value()
        #analyzer.plot_positions_per_asset_separate(assets)

    @staticmethod
    def QuadraticAscend():
        # Load asset data
        assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_over20years")

        # Convert to Polars for speedup
        assetspl: Dict[str, AssetDataPolars] = {}
        for ticker, asset in assets.items():
            assetspl[ticker]= AssetDataService.to_polars(asset)

        # Define strategy
        initialCash=10000.0
        strategy = StratQuadraticAscendRanked(num_months = 1, num_choices= 1)

        # Set up simulation
        simulation = SimulatePortfolio(
            portfolio = Portfolio(cash = initialCash),
            strategy=strategy,
            assets=assetspl,
            startDate=pd.Timestamp(2005,1,4),
            endDate=pd.Timestamp(2024,10,4),
        )

        # Run simulation
        simulation.run()

        # Analyze results
        analyzer = ResultAnalyzer(simulation.portfolio)
        analyzer.plot_portfolio_value()
        #analyzer.plot_positions_per_asset_separate(assets)

    @staticmethod
    def SimCurveML():
        assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_over20years")

        # Define strategy
        initialCash=10000.0
        strategy = StratCurvePrediction(num_months = 1,
                                        modelPath = "src/predictionModule/bin",
                                        modelName= "curveML_snp500_10to20")

        # Set up simulation
        simulation = SimulatePortfolio(
            portfolio = Portfolio(cash = initialCash),
            strategy=strategy,
            assets=assets,
            startDate=pd.Timestamp(2010,1,4),
            endDate=pd.Timestamp(2020,1,4),
        )

        # Run simulation
        simulation.run()

        # Analyze results
        analyzer = ResultAnalyzer(simulation.portfolio)
        analyzer.plot_portfolio_value()
        #analyzer.plot_positions_per_asset_separate(assets)