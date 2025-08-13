import pandas as pd
from datetime import datetime

from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.stockGroupsService.Checks import Checks

import logging
logger = logging.getLogger(__name__)

class GroupOHLCVOver5Years(IGroup):
    def groupName(self) -> str:
        return "group_regOHLCV_over5years"

    def checkAsset(self, asset: AssetData) -> bool:
        now = pd.to_datetime(datetime.now())
        if not Checks.checkOverYear(asset=asset, year=now.year - 5):
            return False
        
        if not Checks.is_regular_ohlcv(asset=asset):
            return False

        return True
