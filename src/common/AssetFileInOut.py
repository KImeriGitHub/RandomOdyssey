import os
import pickle

from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService

class AssetFileInOut:
    def __init__(self, directoryPath: str):
        """Initialize the class with a file path."""
        self.directoryPath = directoryPath

    def saveToFile(self, ad: AssetData):
        """Save a dataclass instance to a file using pickle."""

        assetdict = AssetDataService.to_dict(ad)

        with open(os.path.join(self.directoryPath, ad.ticker +".pkl"), 'wb') as f:
            pickle.dump(assetdict, f)


    def loadFromFile(self, tickername: str) -> AssetData:
        """Load and deserialize an instance of a dataclass from a file using pickle."""

        # Read from the file and unpack
        with open(os.path.join(self.directoryPath, tickername +".pkl"), 'rb') as f:
            assetdictread = pickle.load(f)
        
        # Create an instance of the dataclass using the unpacked dictionary
        return AssetDataService.from_dict(assetdictread)
