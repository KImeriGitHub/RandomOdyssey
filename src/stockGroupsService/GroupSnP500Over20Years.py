import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPandas as DPd

class GroupSnP500Over20Years(IGroup):
  snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]

  def groupName(self) -> str:
   return "group_snp500_over20years"

  def checkAsset(self, asset: AssetData) -> bool:
    adf: pd.DataFrame = asset.shareprice
    first_date: pd.Timestamp = adf.index.min()
    max_date: pd.Timestamp = adf.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=adf.index.tz)
    idx2005 = DPd(adf).getNextLowerOrEqualIndex(pd.Timestamp(year=2005, month=1, day=7, tz='UTC'))
    idx2025 = DPd(adf).getNextLowerOrEqualIndex(pd.Timestamp(year=2024, month=12, day=13, tz='UTC'))

    return ((current_date - first_date).days >= 20 * 366.0) \
      and (self.snp500tickers.__contains__(asset.ticker)) \
      and ((current_date - max_date).days < 60)\
      and (idx2025-idx2005 >= 255*19)