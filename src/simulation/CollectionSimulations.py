from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from src.strategy.StratLinearAscendRanked import StratLinearAscendRanked
from src.strategy.StratFreefallAndHighVol import StratFreefallAndHighVol
from src.strategy.StratQuadraticAscendRanked import StratQuadraticAscendRanked
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
        strategy = StratBuyAndHold(targetTickers=['AAPL'], portfolio=portfolio)

        # Set up simulation
        initialCash=10000.0
        portfolio = Portfolio(cash = initialCash)
        simulation = SimulatePortfolio(
            portfolio = portfolio,
            strategy=strategy,
            assets=[assetG, assetA, assetM],
            startDate=pd.Timestamp(2010,1,1, tz="UTC"),
            endDate=pd.Timestamp(2020,1,1, tz="UTC"),
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
        portfolio = Portfolio(cash = initialCash)
        strategy = StratLinearAscendRanked(portfolio=portfolio, num_months = 1, num_choices= 1)

        # Set up simulation
        simulation = SimulatePortfolio(
            portfolio = portfolio,
            strategy=strategy,
            assets=assetspl,
            startDate=pd.Timestamp(2005,1,4, tz="UTC"),
            endDate=pd.Timestamp(2024,10,4, tz="UTC"),
        )

        # Run simulation
        simulation.run()

        # Analyze results
        analyzer = ResultAnalyzer(simulation.portfolio)
        analyzer.plot_portfolio_value()
        #analyzer.plot_positions_per_asset_separate(assets)

    @staticmethod
    def QuadraticAscend(assets: Dict, 
                        stoplossratio: float = 0.9,
                        num_months = 1,
                        num_months_var = 6,
                        num_choices = 1):

        # Define strategy
        initialCash=10000.0
        portfolio = Portfolio(cash = initialCash)
        strategy = StratQuadraticAscendRanked(
            portfolio=portfolio,
            num_months = num_months,
            num_months_var=num_months_var,
            num_choices= num_choices, 
            stoplossratio = stoplossratio,
            printBuySell=True
        )

        # Set up simulation
        simulation = SimulatePortfolio(
            portfolio = portfolio,
            strategy=strategy,
            assets=assets,
            startDate=pd.Timestamp(2006,1,4, tz="UTC"),
            endDate=pd.Timestamp(2024,10,16, tz="UTC")
        )

        # Run simulation
        simulation.run()

        # Analyze results
        analyzer = ResultAnalyzer(portfolio)
        #analyzer.plot_portfolio_value()
        #analyzer.plot_positions_per_asset_separate(assets)
        print(f"Stoplossratio: {stoplossratio}, EndValue: {portfolio.valueOverTime[-1][1]}")

        del strategy
        del simulation
        
    @staticmethod
    def FreefallAndHighVol(
            assets: Dict, 
            num_choices = 1
        ):
        
        # Define strategy
        initialCash=10000.0
        portfolio = Portfolio(cash = initialCash)
        strategy = StratFreefallAndHighVol(
            num_choices = num_choices, 
            portfolio=portfolio,
            assets=assets)

        # Set up simulation
        startDate=pd.Timestamp(2006,1,4, tz="UTC")
        endDate=pd.Timestamp(2024,1,4, tz="UTC")
        simulation = SimulatePortfolio(
            portfolio = portfolio,
            strategy=strategy,
            assets=assets,
            startDate=startDate,
            endDate=endDate,
        )

        # Run simulation
        simulation.run()

        # Analyze results
        analyzer = ResultAnalyzer(simulation.portfolio)
        analyzer.plot_portfolio_value()
        #analyzer.plot_positions_per_asset_separate(assets)
        
        fullInc = portfolio.valueOverTime[-1][1]/initialCash
        print(f"Total Increase: {fullInc}")
        print(f"Cumulative Annual Growth Factor: {fullInc ** (1/(endDate.year-startDate.year))}")
        return fullInc