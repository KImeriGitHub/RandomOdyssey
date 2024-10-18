# StratLinearAscendRanked_18Sep24
## setup
__stoplossRatio = 0.92
'''
assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_over20years")
initialCash=10000.0
strategy = StratLinearAscendRanked(num_months = 1, num_choices= 1
simulation = SimulatePortfolio(
    portfolio = Portfolio(cash = initialCash),
    strategy=strategy,
    assets=assets,
    startDate=pd.Timestamp(2005,1,4),
    endDate=pd.Timestamp(2024,10,11),
)
'''
## result
>150k