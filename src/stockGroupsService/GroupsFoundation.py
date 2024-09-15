from abc import ABC, abstractmethod
from typing import List
import os

from common.YamlTickerInOut import YamlTickerInOut
from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetData import AssetData
from src.stockGroupsService.Group_Over20Years import Group_Over20Years
from src.stockGroupsService.Group_Swiss import Group_Swiss

class GroupsFoundation(ABC):
    def __init__(self, databasePath: str, stockGroupPath: str):
        self.databasePath = databasePath
        self.stockGroupPath = stockGroupPath

        self.allGroups = List[GroupsFoundation] = [
            Group_Over20Years(self.databasePath, self.stockGroupPath),
            Group_Swiss(self.databasePath, self.stockGroupPath)
        ]

    def checkall(self, groupList: List[List], asset: AssetData):
        for index, g in enumerate(self.allGroups):
            if isinstance(g, GroupsFoundation) and g.checkAsset(asset):
                groupList[index].append(g.groupName())

    def generateYaml(self):
        if not os.path.isfile(os.path.join(self.stockGroupPath, "group_all.yaml")):
            raise FileNotFoundError("The file 'group_all.yaml' is not given.")
        
        allStocksList = YamlTickerInOut(self.stockGroupPath).loadFromFile("group_all.yaml")

        groupList: List[List] = [[] for _ in range(self.allGroups)]
        for t in allStocksList:
            asset: AssetData = AssetFileInOut(self.databasePath).loadFromFile(t)
            print(asset.ticker)
            self.checkall(groupList, asset)

        for index, g in enumerate(self.allGroups):
            YamlTickerInOut(self.stockGroupPath).saveToFile(groupList[index], self.groupName())

    @abstractmethod
    def checkAsset(self, asset: AssetData) -> bool:
        pass

    @abstractmethod
    def groupName(self) -> str:
        pass