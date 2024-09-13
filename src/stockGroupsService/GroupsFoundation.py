from abc import ABC, abstractmethod
import yaml # the library is pyyaml not yaml
import os

from common.YamlTickerInOut import YamlTickerInOut
from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetData import AssetData


class GroupsFoundation(ABC):
    def __init__(self, databasePath: str, stockGroupPath: str):
        self.databasePath = databasePath
        self.stockGroupPath = stockGroupPath

    def generateYaml(self):
        if not os.path.isfile(os.path.join(self.stockGroupPath, "group_all.yaml")):
            raise FileNotFoundError("The file 'group_all.yaml' is not given.")
        
        allStocksList = YamlTickerInOut(self.stockGroupPath).loadFromFile("group_all.yaml")

        groupList: list = []
        for ticker in allStocksList:
            asset: AssetData = AssetFileInOut(self.databasePath).loadFromFile(ticker)
            if self.checkAsset(asset):
                groupList.append(asset.ticker)

        YamlTickerInOut(self.stockGroupPath).saveToFile(groupList, self.groupName())

    @abstractmethod
    def checkAsset(self, asset: AssetData) -> bool:
        pass

    @abstractmethod
    def groupName(self) -> str:
        pass