from datetime import datetime
import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

class GroupOver20Years(IGroup):
    def groupName(self) -> str:
        return "group_over20years"

    def checkAsset(self, asset: AssetData) -> bool:
        adf: pd.DataFrame = asset.shareprice
        first_date: datetime = adf.index.min()
        current_date: datetime = adf.index.max()
        return (current_date - first_date).days >= 20 * 365.25
