from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.stockGroupsService.Checks import Checks

import logging
logger = logging.getLogger(__name__)

class GroupFinanTo2016(IGroup):
    def groupName(self) -> str:
      return "group_finanTo2016" 

    def checkAsset(self, asset: AssetData) -> bool:
        if not Checks.checkFinanTo(asset=asset, year=2016):
            return False
        
        if not Checks.checkOverYear(asset=asset, year=2011):
            return False
        
        return True