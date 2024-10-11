from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from src.strategy.StratLinearAscend import StratLinearAscend
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.Portfolio import Portfolio

import pandas as pd

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
        tickers = ['GOOGL', 'AAPL', 'MSFT', 'IRM', 'T', 
                    'KO', 'AMZN', 'NVO', 'NVDA', 'HRB', 
                    "WARN.SW", "HBLN.SW", "GRKP.SW", "ABBN.SW", "GF.SW"]
        #tickers = ['GOOGL', 'AAPL', 'MSFT']
        #tickers = YamlTickerInOut("src/stockGroups").loadFromFile("group_snp500_over20years")
        assets={}
        for i, ticker in enumerate(tickers):
            assets[ticker] = AssetFileInOut("src/database").loadFromFile(ticker)
            if i % (len(tickers) // 10) == 0:
                print(f"{i / len(tickers):.0%} loaded.")

        # Define strategy
        initialCash=10000.0
        strategy = StratLinearAscend(num_months = 1, num_choices= 1)

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
        analyzer.plot_positions_per_asset_separate(assets)