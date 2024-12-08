import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut

class GroupSnP500FinanTo2011(IGroup):
  snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]

  def groupName(self) -> str:
   return "group_snp500_finanTo2011"

  def checkAsset(self, asset: AssetData) -> bool:
    if asset.financials_quarterly.columns.__contains__('fiscalDateEnding'):
      columnName = 'fiscalDateEnding'
    elif asset.financials_quarterly.columns.__contains__('Date'): 
      columnName = 'Date'
    else:
      return False
    
    adf: pd.DataFrame = asset.shareprice
    first_date: pd.Timestamp = adf.index.min()
    max_date: pd.Timestamp = adf.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=adf.index.tz)
    df_year = adf[asset.financials_quarterly[columnName].dt.year == 2011]
    return ((current_date - first_date).days >= 20 * 366.0) \
      and (self.snp500tickers.__contains__(asset.ticker)) \
      and ((current_date - max_date).days < 60) \
      and df_year.notna().all(axis=1)