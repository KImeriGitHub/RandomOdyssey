import pandas as pd
from datetime import datetime

from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.stockGroupsService.Checks import Checks

import logging
logger = logging.getLogger(__name__)

class GroupFinanTo2011(IGroup):

    def groupName(self) -> str:
        return "group_finanTo2011"

    def checkAsset(self, asset: AssetData) -> bool:
        if not Checks.checkFinanTo(asset=asset, year=2011):
            return False
        
        if not Checks.checkOverYear(asset=asset, year=2008):
            return False
  
        return True