import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.stockGroupsService.GroupSwiss import GroupSwiss
from src.stockGroupsService.GroupOver20Years import GroupOver20Years

class GroupSwissOver20Years(IGroup):
    def groupName(self) -> str:
        return "group_swiss_over20years"

    def checkAsset(self, asset: AssetData) -> bool:
        if not GroupOver20Years.checkAsset(self, asset):
            return False
        
        if not GroupSwiss.checkAsset(self, asset):
            return False
        
        return True
