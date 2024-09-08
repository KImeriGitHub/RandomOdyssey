import os
from src.databaseService.FileInOut import FileInOut
from src.common.AssetDataService import AssetDataService
from src.common.AssetData import AssetData

def test_FileExists_true():
    fileOut = FileInOut("unittests/database")

    asset = AssetDataService.defaultInstance()
    asset.ticker="test"
    fileOut.saveToFile(asset)

    destPath = os.path.join("unittests", "database", "test.pkl")
    assert os.path.exists(destPath), "File should exist"

    # After the test, clean up the file
    if os.path.exists(destPath):
        os.remove(destPath)
        pass