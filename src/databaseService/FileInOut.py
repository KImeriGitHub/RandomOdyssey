import os
import msgpack
import json
from dataclasses import asdict, dataclass

from src.common.AssetData import AssetData

class FileInOut:
    def __init__(self, directoryPath: str):
        """Initialize the class with a file path."""
        self.directoryPath = directoryPath

    def saveToFile(self, ad: AssetData):
        """Save a dataclass instance to a file using msgpack."""

        # Convert dataclass to a dictionary for msgpack compatibility
        ad_dict = ad.to_dict()
        
        # Write the serialized object to a file
        with open(os.path.join(self.directoryPath, ad.ticker +".bin"), 'w') as f:
            #packed_data = msgpack.packb(ad_dict)
            f.write(str(ad_dict))
            #json.dump(ad_dict, f)


    def loadFromFile(self, ad: AssetData):
        """Load and deserialize an instance of a dataclass from a file using msgpack."""
        
        # Read from the file and unpack
        with open(os.path.join(self.directoryPath, ad.ticker +".bin"), 'rb') as f:
            packed_data = f.read()
            ad_dict = msgpack.unpackb(packed_data)
        
        # Create an instance of the dataclass using the unpacked dictionary
        return ad(**ad_dict)
