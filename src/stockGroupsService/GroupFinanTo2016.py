from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.stockGroupsService.GroupFinanTo2011 import GroupFinanTo2011

import logging
logger = logging.getLogger(__name__)

class GroupFinanTo2016(IGroup):
    def groupName(self) -> str:
      return "group_finanTo2016" 

    def checkAsset(self, asset: AssetData) -> bool:
        if not GroupFinanTo2011.checkAsset(self, asset, 2016):
            return False
        
        return True
