import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

class GroupDebug(IGroup):
  tickers = [
    "CSX",
    "EXC",
    "NVDA",
    "ADBE",
    "AMZN",
    "AMD",
    "AMGN",
    "ADI",
    "ANSS",
    "AAPL",
    "ADP",
    "BKNG",
    "CDNS",
    "CSCO",
    "CTSH",
    "CMCSA",
    "CSX",
    "EA",
    "EXC",
    "GILD",
    "IDXX",
    "INTC",
    "INTU",
    "ISRG",
    "KLAC",
    "LRCX",
    "MAR",
    "MU",
    "MSFT",
    "MDLZ",
    "NFLX",
    "NVDA",
    "QCOM",
    "REGN",
    "ROST",
    "SBUX",
    "SNPS",
    "TXN",
    "ALGN",
  ]

  def groupName(self) -> str:
   return "group_debug"

  def checkAsset(self, asset: AssetData) -> bool:
    
    adf: pd.DataFrame = asset.shareprice
    max_date: pd.Timestamp = adf.index.max()
    current_date: pd.Timestamp = pd.Timestamp.now(tz=adf.index.tz)
    return (self.tickers.__contains__(asset.ticker)) \
      and ((current_date - max_date).days < 60)