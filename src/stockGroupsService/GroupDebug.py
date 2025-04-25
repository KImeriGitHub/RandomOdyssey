import pandas as pd
from datetime import datetime
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

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
        max_date: pd.Timestamp = pd.to_datetime(asset.shareprice["Date"]).max()
        today: pd.Timestamp = pd.to_datetime(datetime.now())
        
        if max_date < today - pd.Timedelta(days=20):
            logging.warning(f"Last date in shareprice is too old: {max_date}")
            return False
        return True