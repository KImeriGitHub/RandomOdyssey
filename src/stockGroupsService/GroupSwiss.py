from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

class GroupSwiss(IGroup):
    def groupName(self) -> str:
        return "group_swiss"

    def checkAsset(self, asset: AssetData) -> bool:
        return asset.ticker.lower().endswith(".sw")
