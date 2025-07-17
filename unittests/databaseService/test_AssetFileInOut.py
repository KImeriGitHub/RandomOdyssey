import os
import pandas as pd
import polars as pl
from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetDataService import AssetDataService

def test_FileInOut_FileExists_true():
    fileOut = AssetFileInOut("unittests/database")

    asset = AssetDataService.defaultInstance()
    asset.ticker="test"
    fileOut.saveToFile(asset)

    destPath = os.path.join("unittests", "database", "test.pkl")
    assert os.path.exists(destPath), "File should exist"

    # After the test, clean up the file
    if os.path.exists(destPath):
        os.remove(destPath)

def test_FileInOut_FileLoadedSameMOCKED_true():
    fileOut = AssetFileInOut("unittests/database")

    asset = AssetDataService.defaultInstance()
    asset.ticker="test"

    data = {
    'Date': ['1996-02-01 00:00:00-05:00', '1996-02-02 00:00:00-05:00', '1996-02-05 00:00:00-05:00', 
             '1996-02-06 00:00:00-05:00', '1996-02-07 00:00:00-05:00'],
    'Open': [1.236482, 1.226971, 1.236482, 1.217460, 1.217460],
    'High': [1.236482, 1.236482, 1.236482, 1.236482, 1.236482],
    'Low': [1.217459, 1.217460, 1.217460, 1.217460, 1.217460],
    'Close': [1.226971, 1.236482, 1.217460, 1.217460, 1.217460],
    'AdjClose': [1.126971, 1.136482, 1.117460, 1.117460, 1.017460],
    }

    # Convert the dictionary to a pandas dataframe
    shareprice = pd.DataFrame(data)
    shareprice['Date'] = pd.to_datetime(shareprice['Date'])  # Convert the 'Date' column to datetime

    # Create the pandas Series from the given data
    revenue_data = {
        'Date': ['2024-06-30', '2024-03-31', '2023-12-31', '2023-09-30', '2023-06-30', '2023-03-31', '2022-12-31'],
        'Total Revenue': [1534409000.0, 1476863000.0, 1419829000.0, 1388175000.0, 1357936000.0, None, None]
    }

    # Convert to pandas Series, with Date as the index
    revenue_series = pd.Series(revenue_data['Total Revenue'], index=pd.to_datetime(revenue_data['Date']), name="Total Revenue")

    asset.shareprice = shareprice
    asset.revenue = revenue_series

    fileOut.saveToFile(asset)

    destPath = os.path.join("unittests", "database", "test.pkl")
    assert os.path.exists(destPath), "File should exist"

    loadedAsset = fileOut.loadFromFile("test")

    assert asset.ticker == loadedAsset.ticker, "Ticker not same"
    assert asset.isin == loadedAsset.isin, "isin not same"
    assert asset.shareprice.equals(loadedAsset.shareprice), "shareprice not same"
    assert asset.adjClosePrice.equals(loadedAsset.adjClosePrice), "volume not same"
    assert asset.volume.equals(loadedAsset.volume), "volume not same"
    assert asset.dividends.equals(loadedAsset.dividends), "dividends not same"
    assert asset.splits.equals(loadedAsset.splits), "splits not same"
    assert asset.about == loadedAsset.about, "about not same"
    assert asset.revenue.equals(loadedAsset.revenue), "revenue not same"
    assert asset.EBITDA.equals(loadedAsset.EBITDA), "EBITDA not same"
    assert asset.basicEPS.equals(loadedAsset.basicEPS), "basicEPS not same"

    # After the test, clean up the file
    if os.path.exists(destPath):
        os.remove(destPath)