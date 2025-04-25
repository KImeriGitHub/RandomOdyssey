import pandas as pd
from datetime import datetime

from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut

import logging
logger = logging.getLogger(__name__)

class GroupSnP500(IGroup):
    snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]
  
    def groupName(self) -> str:
        return "group_snp500"
  
    def checkAsset(self, asset: AssetData) -> bool:
        max_date: pd.Timestamp = pd.to_datetime(asset.shareprice["Date"]).max()
        today: pd.Timestamp = pd.to_datetime(datetime.now())
        
        if not asset.ticker in self.snp500tickers:
            return False
        
        if max_date < today - pd.Timedelta(days=20):
            logging.warning(f"Last date in shareprice is too old: {max_date}")
            return False
        
        return True