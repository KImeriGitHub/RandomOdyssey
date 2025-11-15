from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.stockGroupsService.Checks import Checks

import logging
logger = logging.getLogger(__name__)

class GroupOHLCVTo2014(IGroup):
    def groupName(self) -> str:
        return "group_regOHLCV_to2014"

    def checkAsset(self, asset: AssetData) -> bool:
        if not Checks.checkOverYear(asset=asset, year=2014):
            return False
        
        if not Checks.is_regular_ohlcv(asset=asset):
            return False

        return True
