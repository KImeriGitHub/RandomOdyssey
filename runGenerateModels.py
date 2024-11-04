from src.predictionModule.CollectionModels import CollectionModels
from src.common.AssetFileInOut import AssetFileInOut

from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
from typing import Dict

assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_over20years")
#assets = {}
#tickers = ['GOOGL', 'AAPL', 'MSFT']
#for ticker in tickers:
#    assets[ticker] = AssetFileInOut("src/database").loadFromFile(ticker)
# Convert to Polars for speedup
assetspl: Dict[str, AssetDataPolars] = {}
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)

if __name__ == "__main__":
    CollectionModels.fourierML_snp500_10to20(assets=assetspl)