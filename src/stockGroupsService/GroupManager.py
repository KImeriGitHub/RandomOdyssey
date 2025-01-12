import os
from typing import List, Dict
import numpy as np
from src.common.YamlTickerInOut import YamlTickerInOut
from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

class GroupManager:
    def __init__(self, databasePath: str, stockGroupPath: str, groupClasses: List[IGroup]):
        self.databasePath = databasePath
        self.stockGroupPath = stockGroupPath
        self.groupClasses = groupClasses

    def generateGroups(self):
        group_all_path = os.path.join(self.stockGroupPath, "group_all.yaml")
        if not os.path.isfile(group_all_path):
            raise FileNotFoundError("The file 'group_all.yaml' is not given.")

        all_stocks_list = YamlTickerInOut(self.stockGroupPath).loadFromFile("group_all.yaml")
        all_stocks_list = np.unique(all_stocks_list)
        # Initialize group lists
        group_lists = {criterion.groupName(): [] for criterion in self.groupClasses}
        assets: Dict[str, AssetData] = {}
        for ticker in all_stocks_list:
            assets[ticker] = AssetFileInOut(self.databasePath).loadFromFile(ticker)
            print(f"Processing asset: {assets[ticker].ticker}")
            for criterion in self.groupClasses:
                if criterion.checkAsset(assets[ticker]):
                    group_lists[criterion.groupName()].append(assets[ticker].ticker)

        # Save group lists to YAML files
        for groupName, grouptickers in group_lists.items():
            YamlTickerInOut(self.stockGroupPath).saveToFile(grouptickers, groupName)
            assetsInGroup: Dict[str, AssetData] = {}
            for ticker in grouptickers:
                assetsInGroup[ticker] = assets[ticker]

            AssetFileInOut(os.path.join(self.stockGroupPath, "bin")).saveDictToFile(assetsInGroup,groupName)