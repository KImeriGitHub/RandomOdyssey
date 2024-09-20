from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut

import pandas as pd

def runStrategy():
    # Load asset data
    asset = AssetFileInOut("src/database").loadFromFile('GOOG')  # Example ticker

    # Define strategy
    strategy = StratBuyAndHold(target_ticker='GOOG')

    # Set up simulation
    simulation = SimulatePortfolio(
        initialCash=10000,
        strategy=strategy,
        assets=[asset],
        startDate=pd.Timestamp('2010-01-01'),
        endDate=pd.Timestamp('2020-01-01'),
    )

    # Run simulation
    simulation.run()

    # Analyze results
    analyzer = ResultAnalyzer(simulation.portfolio)
    analyzer.plot_portfolio_value()