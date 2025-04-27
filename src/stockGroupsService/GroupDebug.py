import pandas as pd
from datetime import datetime
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

from src.stockGroupsService.GroupFinanTo2011 import GroupFinanTo2011
from src.stockGroupsService.GroupOver20Years import GroupOver20Years

import logging
logger = logging.getLogger(__name__)

class GroupDebug(IGroup):
    tickers = [
        "AAPL",
        "ADBE",
        "ADI",
        "ADP",
        "ALGN",
        "AMGN",
        "AMZN",
        "ANSS",
        "BKNG",
        "CDNS",
        "CSCO",
        "CSX",
        "CTSH",
        "EXC",
        "GILD",
        "IDXX",
        "INTC",
        "ISRG",
        "KLAC",
        "MAR",
        "MDLZ",
        "MSFT",
        "MU",
        "NFLX",
        "NVDA",
        "QCOM",
        "REGN",
        "SBUX",
        "TXN"
    ]

    def groupName(self) -> str:
        return "group_debug"

    def checkAsset(self, asset: AssetData) -> bool:
        if not GroupFinanTo2011.checkAsset(self, asset):
            return False
        
        if not GroupOver20Years.checkAsset(self, asset):
            return False
        
        if asset.ticker not in self.tickers:
            return False
        
        return True