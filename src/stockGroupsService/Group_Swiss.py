import pandas as pd
from datetime import datetime

from src.stockGroupsService.GroupsFoundation import GroupsFoundation
from src.common.AssetData import AssetData

class Group_Swiss(GroupsFoundation):
    def __init__(self, databasePath: str, stockGroupPath: str):
        super().__init__(databasePath = databasePath, stockGroupPath = stockGroupPath)

    def groupName(self) -> str:
        return "group_swiss"

    def checkAsset(self, asset: AssetData) -> bool:
        # Check whether the stock is swiss
        return asset.ticker.lower().endswith(".sw")