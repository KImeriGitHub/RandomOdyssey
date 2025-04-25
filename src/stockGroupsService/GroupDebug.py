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
        "CSX",
        "EA",
        "EXC",
        "GILD",
        "IDXX",
        "INTC",
        "INTU",
        "ISRG",
        "KLAC",
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
        if GroupFinanTo2011.checkAsset(self, asset) == False:
            return False
        
        if GroupOver20Years.checkAsset(self, asset) == False:
            return False
        
        if asset.ticker not in self.tickers:
            return False
        
        return True