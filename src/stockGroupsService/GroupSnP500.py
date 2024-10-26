import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut

class GroupSnP500(IGroup):
  snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]

  def groupName(self) -> str:
   return "group_snp500"

  def checkAsset(self, asset: AssetData) -> bool:
    max_date: pd.Timestamp = asset.shareprice.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=asset.shareprice.index.tz)
    return self.snp500tickers.__contains__(asset.ticker) \
      and ((current_date - max_date).days < 60)