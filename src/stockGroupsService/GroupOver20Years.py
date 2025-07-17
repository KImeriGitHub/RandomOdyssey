import pandas as pd
from datetime import datetime

from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.stockGroupsService.Checks import Checks

import logging
logger = logging.getLogger(__name__)

class GroupOver20Years(IGroup):
    def groupName(self) -> str:
        return "group_over20years"

    def checkAsset(self, asset: AssetData) -> bool:
        if not Checks.checkOverYear(asset=asset, year=2004):
            return False
    
        return True
