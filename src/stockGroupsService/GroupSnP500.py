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
        if not asset.ticker in self.snp500tickers:
            return False
        
        return True