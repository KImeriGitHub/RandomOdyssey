# RandomOdyssey\src\common\AssetFileInOut.py
import os
import pandas as pd
from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService

class AssetFileInOut:
    def __init__(self, directoryPath: str):
        self.directoryPath = directoryPath

    def saveToFile(self, ad: AssetData):
        assetdict = AssetDataService.to_dict(ad)
        file_path = os.path.join(self.directoryPath, f"{ad.ticker}.pkl")
        pd.to_pickle(assetdict, file_path)

    def loadFromFile(self, tickername: str) -> AssetData:
        file_path = os.path.join(self.directoryPath, f"{tickername}.pkl")
        assetdictread = pd.read_pickle(file_path)
        return AssetDataService.from_dict(assetdictread)
