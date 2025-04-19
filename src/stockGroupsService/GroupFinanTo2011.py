import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPandas as DPd

class GroupFinanTo2011(IGroup):

  def groupName(self) -> str:
   return "group_finanTo2011"

  def checkAsset(self, asset: AssetData) -> bool:
    quarterly_entries = 4*13
    annual_entries = 13
    
    if asset.financials_quarterly is None:
      return False
    if asset.financials_annually is None:
      return False
    if not asset.financials_quarterly.columns.__contains__('fiscalDateEnding'):
      return False
    if not asset.financials_annually.columns.__contains__('fiscalDateEnding'):
      return False
    if len(asset.financials_annually) < annual_entries:
      return False
    if len(asset.financials_quarterly) < quarterly_entries:
      return False
    if asset.financials_quarterly["reportedEPS"].tail(quarterly_entries).isnull().sum() > 0:
      return False
    
    # Check if the asset has no empty entries in the quarterly financials in the columns quarterly_columns and annual financials in the columns annual_columns for the last n entries
    buffer_quar = 0
    buffer_ann = 0
    if asset.financials_quarterly[GroupFinanTo2011.quarterly_columns].tail(quarterly_entries).isnull().sum().sum() > buffer_quar:
      return False
    if asset.financials_annually[GroupFinanTo2011.annual_columns].tail(annual_entries).isnull().sum().sum() > buffer_ann:
      return False
    if all([asset.financials_quarterly[col].tail(quarterly_entries).isnull().sum() > 0 for col in GroupFinanTo2011.quarterly_columns]):
      return False
    if all([asset.financials_annually[col].tail(annual_entries).isnull().sum() > 0 for col in GroupFinanTo2011.annual_columns]):
      return False
    
    adf: pd.DataFrame = asset.shareprice
    first_date: pd.Timestamp = adf.index.min()
    max_date: pd.Timestamp = adf.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=adf.index.tz)
    #df_year = asset.financials_quarterly[asset.financials_quarterly['fiscalDateEnding'].dt.year == 2011]
    idx2008 = DPd(adf).getNextLowerOrEqualIndex(pd.Timestamp(year=2008, month=1, day=7, tz='UTC'))
    idx2025 = DPd(adf).getNextLowerOrEqualIndex(pd.Timestamp(year=2024, month=12, day=13, tz='UTC'))

    return (
      ((current_date - first_date).days >= 20 * 366.0) 
      and ((current_date - max_date).days < 60)
      and (idx2025-idx2008 >= 255*16) )
  
  compact_columns = [
    'fiscalDateEnding',
    'reportedEPS',
    'grossProfit',
    'totalRevenue',
    'ebitda',
    'totalAssets',
  ]

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
    'totalShareholderEquity',
    'commonStockSharesOutstanding',
    'operatingCashflow',
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
    'totalShareholderEquity',
    'operatingCashflow',
  ]