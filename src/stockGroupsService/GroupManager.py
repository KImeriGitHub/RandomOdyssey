import os
from typing import List
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

class GroupManager:
    def __init__(self, databasePath: str, stockGroupPath: str, groupCriteria: List[IGroup]):
        self.databasePath = databasePath
        self.stockGroupPath = stockGroupPath
        self.groupCriteria = groupCriteria

    def generateGroups(self):
        group_all_path = os.path.join(self.stockGroupPath, "group_all.yaml")
        if not os.path.isfile(group_all_path):
            raise FileNotFoundError("The file 'group_all.yaml' is not given.")

        all_stocks_list = YamlTickerInOut(self.stockGroupPath).loadFromFile("group_all.yaml")

        # Initialize group lists
        group_lists = {criterion.groupName(): [] for criterion in self.groupCriteria}

        for ticker in all_stocks_list:
            asset = AssetFileInOut(self.databasePath).loadFromFile(ticker)
            print(f"Processing asset: {asset.ticker}")
            for criterion in self.groupCriteria:
                if criterion.checkAsset(asset):
                    group_lists[criterion.groupName()].append(asset.ticker)

        # Save group lists to YAML files
        for groupName, tickers in group_lists.items():
            YamlTickerInOut(self.stockGroupPath).saveToFile(tickers, groupName)