import os
import yaml # the library is pyyaml not yaml

from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService

class YamlTickerInOut:
    def __init__(self, directoryPath: str):
        """Initialize the class with a file path."""
        self.directoryPath = directoryPath

    def saveToFile(self, var, filename: str):
        # Check if the filename ends with '.yaml', and if not, add it
        if not filename.lower().endswith('.yaml'):
            filename += '.yaml'
        
        with open(os.path.join(self.directoryPath, filename), 'w') as file:
            yaml.dump(var, file, default_flow_style=False)

    def loadFromFile(self, filename: str) -> any:
        if not filename.lower().endswith('.yaml'):
            filename += '.yaml'
        file_path = os.path.join(self.directoryPath, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        with open(file_path, 'r') as file:
            tickersDict = yaml.safe_load(file) or {}
        return tickersDict
