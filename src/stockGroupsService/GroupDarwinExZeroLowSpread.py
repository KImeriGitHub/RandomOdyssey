import pandas as pd
from datetime import datetime

from src.common.AssetData import AssetData
from src.stockGroupsService.Checks import Checks
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut

import logging
logger = logging.getLogger(__name__)

class GroupDarwinExZeroLowSpread(IGroup):
    dez_low_spread_tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("darwinexzero_low_spread.yaml")["darwinexzerotickers"]

    def groupName(self) -> str:
        return "group_dez_lowspread"

    def checkAsset(self, asset: AssetData) -> bool:
        if not asset.ticker in self.dez_low_spread_tickers:
            return False
        
        if not Checks.checkFinanTo(asset=asset, year=2011):
            return False
        
        if not Checks.checkOverYear(asset=asset, year=2008):
            return False
        
        return True