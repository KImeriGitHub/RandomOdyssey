import pandas as pd
from datetime import datetime

from src.common.AssetData import AssetData
from src.stockGroupsService.Checks import Checks
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut

import logging
logger = logging.getLogger(__name__)

class GroupMT5(IGroup):
    mt5tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("mt5.yaml")["mt5tickers"]

    def groupName(self) -> str:
        return "group_mt5"

    def checkAsset(self, asset: AssetData) -> bool:
        if not asset.ticker in self.mt5tickers:
            return False
        
        if not Checks.checkFinanTo(asset=asset, year=2011):
            return False
        
        if not Checks.checkOverYear(asset=asset, year=2008):
            return False
        
        return True