from src.simulation.SimulatePortfolio import SimulatePortfolio
from src.strategy.StratBuyAndHold import StratBuyAndHold
from src.simulation.ResultAnalyzer import ResultAnalyzer
from src.common.AssetFileInOut import AssetFileInOut

import datetime

# Load asset data
fileInOut = AssetFileInOut("src/database")
asset = fileInOut.loadFromFile('AAPL')  # Example ticker

# Define strategy
strategy = StratBuyAndHold(target_ticker='AAPL')

# Set up simulation
simulation = SimulatePortfolio(
    initial_cash=10000,
    strategy=strategy,
    assets=[asset],
    start_date=datetime.datetime(2010, 1, 1),
    end_date=datetime.datetime(2020, 1, 1)
)

# Run simulation
simulation.run()

# Analyze results
analyzer = ResultAnalyzer(simulation.portfolio)
analyzer.plot_portfolio_value()