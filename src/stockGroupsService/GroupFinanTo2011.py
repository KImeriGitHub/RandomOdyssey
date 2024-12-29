import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut

class GroupFinanTo2011(IGroup):

  def groupName(self) -> str:
   return "group_finanTo2011"

  def checkAsset(self, asset: AssetData) -> bool:
    quarterly_entries = 4*13
    annual_entries = 4
    
    if asset.financials_quarterly is None:
      return False
    if asset.financials_annually is None:
      return False
    if not asset.financials_quarterly.columns.__contains__('fiscalDateEnding'):
      return False
    if not asset.financials_annually.columns.__contains__('fiscalDateEnding'):
      return False
    if len(asset.financials_annually) < annual_entries: # Annual data is quite unattainable
      return False
    if len(asset.financials_quarterly) < quarterly_entries:
      return False
    if asset.financials_quarterly["reportedEPS"].tail(quarterly_entries).isnull().sum() > 0:
      return False
    
    # Check if the asset has no empty entries in the quarterly financials in the columns quarterly_columns and annual financials in the columns annual_columns for the last n entries
    buffer_quar = 10
    buffer_ann = 10
    if asset.financials_quarterly[GroupFinanTo2011.quarterly_columns].tail(quarterly_entries).isnull().sum().sum() > buffer_quar:
      return False
    if asset.financials_annually[GroupFinanTo2011.annual_columns].tail(annual_entries).isnull().sum().sum() > buffer_ann:
      return False
    
    adf: pd.DataFrame = asset.shareprice
    first_date: pd.Timestamp = adf.index.min()
    max_date: pd.Timestamp = adf.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=adf.index.tz)
    #df_year = asset.financials_quarterly[asset.financials_quarterly['fiscalDateEnding'].dt.year == 2011]
    return ((current_date - first_date).days >= 20 * 366.0) \
      and ((current_date - max_date).days < 60) 
      
  quarterly_columns = [
    'fiscalDateEnding',
    'reportedDate',
    'reportedEPS',
    'estimatedEPS',
    'surprise',
    'surprisePercentage',
    'reportTime',
    'grossProfit',
    'totalRevenue',
    'ebit',
    'ebitda',
    'totalAssets',
    'totalCurrentLiabilities',
    'operatingCashflow',
    'profitLoss',
  ]
    
  annual_columns = [
    'fiscalDateEnding',
    'reportedEPS',
    'grossProfit',
    'totalRevenue',
    'ebit',
    'ebitda',
    'totalAssets',
    'totalCurrentLiabilities',
    'operatingCashflow',
    'profitLoss',
  ]