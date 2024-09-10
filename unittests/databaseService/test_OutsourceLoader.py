import os
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from src.databaseService.FileInOut import FileInOut
from src.common.AssetDataService import AssetDataService
from src.common.AssetData import AssetData
from src.databaseService.OutsourceLoader import OutsourceLoader

def test_OutsourceLoader_Loadsyfinance():
    outsourceLoader = OutsourceLoader(outsourceOperator="yfinance")

    asset: AssetData = outsourceLoader.load(ticker="irm")

    assert asset.ticker == "IRM", "Ticker not same"
    assert asset.isin == "-", "isin not same"
    assert type(asset.shareprice) == type(pd.DataFrame(None)), "shareprice not DataFrame"
    assert type(asset.volume) == type(pd.Series(None)), "volume not Series"
    assert type(asset.dividends) == type(pd.Series(None)), "dividends not Series"
    assert type(asset.splits) == type(pd.Series(None)), "splits not Series"
    assert type(asset.about) == type({}), "about not dict"
    assert type(asset.revenue) == type(pd.Series(None)), "revenue not Series"
    assert type(asset.EBITDA) == type(pd.Series(None)), "EBITDA not Series"
    assert type(asset.basicEPS) == type(pd.Series(None)), "basicEPS not Series"

    assert is_datetime64_any_dtype(asset.shareprice.index), "The dataframe index is not a date."
    assert is_datetime64_any_dtype(asset.volume.index), "The volume index is not a date."
    assert is_datetime64_any_dtype(asset.dividends.index), "The dividends index is not a date."
    assert is_datetime64_any_dtype(asset.splits.index), "The splits index is not a date."
    assert is_datetime64_any_dtype(asset.revenue.index), "The revenue index is not a date."
    assert is_datetime64_any_dtype(asset.EBITDA.index), "The EBITDA index is not a date."
    assert is_datetime64_any_dtype(asset.basicEPS.index), "The basicEPS index is not a date."

    asset2: AssetData = outsourceLoader.load(ticker = "CH0011432447")

    assert asset2.ticker == "BSLN.SW", "Ticker not same"
    assert asset2.isin == "-", "isin not same"
    assert type(asset2.shareprice) == type(pd.DataFrame(None)), "shareprice not DataFrame"
    assert type(asset2.volume) == type(pd.Series(None)), "volume not Series"
    assert type(asset2.dividends) == type(pd.Series(None)), "dividends not Series"
    assert type(asset2.splits) == type(pd.Series(None)), "splits not Series"
    assert type(asset2.about) == type({}), "about not dict"
    assert type(asset2.revenue) == type(pd.Series(None)), "revenue not Series"
    assert type(asset2.EBITDA) == type(pd.Series(None)), "EBITDA not Series"
    assert type(asset2.basicEPS) == type(pd.Series(None)), "basicEPS not Series"

    asset3: AssetData = outsourceLoader.load(ticker = "goog")
    assert asset3.isin == "ARDEUT116159", "isin not same"

def test_OutsourceLoader_LoadsUndefined_yfinance():
    outsourceLoader = OutsourceLoader(outsourceOperator="yfinance")
    ticker="aaaa"
    try:
        asset: AssetData = outsourceLoader.load(ticker=ticker)
    except:
        print("Exception was raised as expected")
        return
    assert False, "TypeError was not raised"