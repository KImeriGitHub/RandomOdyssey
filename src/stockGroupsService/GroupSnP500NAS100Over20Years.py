import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut

class GroupSnP500NAS100Over20Years(IGroup):
  snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]
  nas100tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("nas100.yaml")["nas100tickers"]

  def groupName(self) -> str:
   return "group_nas100snp500_over20years"

  def checkAsset(self, asset: AssetData) -> bool:
    adf: pd.DataFrame = asset.shareprice
    first_date: pd.Timestamp = adf.index.min()
    max_date: pd.Timestamp = adf.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=adf.index.tz)
    return ((current_date - first_date).days >= 20 * 366.0) \
      and ((self.snp500tickers.__contains__(asset.ticker)) \
        or (self.nas100tickers.__contains__(asset.ticker))) \
      and ((current_date - max_date).days < 60)