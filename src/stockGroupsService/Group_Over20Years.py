import pandas as pd
from datetime import datetime

from src.stockGroupsService.GroupsFoundation import GroupsFoundation
from src.common.AssetData import AssetData

class Group_Over20Years(GroupsFoundation):
    def __init__(self, databasePath: str, stockGroupPath: str):
        super().__init__(databasePath = databasePath, stockGroupPath = stockGroupPath)

    def groupName(self) -> str:
        return "group_over20years"

    def checkAsset(self, asset: AssetData) -> bool:
        adf: pd.DataFrame = asset.shareprice

        first_date: datetime = adf.index.min()
        current_date: datetime = adf.index.max()

        # Check if the difference is at least 20 years
        return (current_date - first_date).days >= 20 * 365.25