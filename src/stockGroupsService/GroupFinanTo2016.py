import pandas as pd
from datetime import datetime, timedelta

from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPandas as DPd
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
