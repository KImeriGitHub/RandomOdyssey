import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

class GroupDebug(IGroup):
  tickers = [
    "GOOG", "MSFT", "AMZN", "TSLA", "AAPL", "FB", "NVDA", "PYPL", 
    "ADBE", "NFLX", "INTC", "CSCO", "CMCSA", "PEP", "COST", "AMGN", 
    "AVGO", "TXN", "QCOM", "GILD", "SBUX", "MDLZ", "INTU", "BKNG", 
    "TMUS", "AMD", "MU", "ADP", "ISRG", "REGN", "FISV", "ATVI", "CSX", 
    "ADI", "ILMN", "ADI", "MELI", "LRCX", "KHC", "EXC", "CTSH", "WBA", 
    "MAR", "EA", "ROST", "KLAC", "IDXX", "ALGN", "ANSS", "CDNS", "SNPS",  
  ]

  def groupName(self) -> str:
   return "group_debug"

  def checkAsset(self, asset: AssetData) -> bool:
    
    adf: pd.DataFrame = asset.shareprice
    max_date: pd.Timestamp = adf.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=adf.index.tz)
    return (self.tickers.__contains__(asset.ticker)) \
      and ((current_date - max_date).days < 60)