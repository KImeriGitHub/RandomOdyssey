import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

class GroupAmericanOver20Years(IGroup):
    def groupName(self) -> str:
        return "group_american_over20years"

    def checkAsset(self, asset: AssetData) -> bool:
        adf: pd.DataFrame = asset.shareprice
        first_date: pd.Timestamp = adf.index.min()
        max_date: pd.Timestamp = adf.index.max()
        current_date: pd.Timestamp = pd.Timestamp.now(tz='UTC')
        return ((current_date - first_date).days >= 20 * 366.0) \
            and not (asset.ticker.__contains__('.')) \
            and ((current_date - max_date).days < 60)