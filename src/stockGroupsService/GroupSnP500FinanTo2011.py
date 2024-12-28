import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut

class GroupSnP500FinanTo2011(IGroup):
  snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]

  def groupName(self) -> str:
   return "group_snp500_finanTo2011"

  def checkAsset(self, asset: AssetData) -> bool:
    if asset.financials_quarterly is None:
      return False
    if asset.financials_annually is None:
      return False
    if not asset.financials_quarterly.columns.__contains__('fiscalDateEnding'):
      return False
    if not asset.financials_annually.columns.__contains__('fiscalDateEnding'):
      return False
    if len(asset.financials_annually) < 4: # Annual data is quite unattainable
      return False
    if len(asset.financials_quarterly) < 4*14:
      return False
    if asset.financials_quarterly["reportedEPS"].tail(4*15).isnull().sum() > 0:
      return False
    
    adf: pd.DataFrame = asset.shareprice
    first_date: pd.Timestamp = adf.index.min()
    max_date: pd.Timestamp = adf.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=adf.index.tz)
    #df_year = asset.financials_quarterly[asset.financials_quarterly['fiscalDateEnding'].dt.year == 2011]
    return ((current_date - first_date).days >= 20 * 366.0) \
      and (self.snp500tickers.__contains__(asset.ticker)) \
      and ((current_date - max_date).days < 60) 