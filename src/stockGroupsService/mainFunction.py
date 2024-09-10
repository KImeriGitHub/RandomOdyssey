import yaml # the library is pyyaml not yaml
import os

from src.common.AssetData import AssetData
from src.common.AssetDataService import AssetDataService
from src.databaseService.FileInOut import FileInOut

# Main function
def mainFunction():
    for subclass in parent_class.__subclasses__():
        instance = subclass()  # Create an instance of the subclass
        instance.display()      # Call the display method for each subclass