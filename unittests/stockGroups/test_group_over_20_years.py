import pandas as pd
from datetime import timedelta
from src.common.AssetData import AssetData
from src.stockGroupsService.GroupOver20Years import GroupOver20Years


def make_shareprice(days: int) -> pd.DataFrame:
    start = pd.Timestamp.now() - timedelta(days=days)
    dates = pd.date_range(start=start, periods=days)
    data = {
        "Date": dates.astype(str),
        "Open": 1.0,
        "High": 1.0,
        "Low": 1.0,
        "Close": 1.0,
        "AdjClose": 1.0,
        "Volume": 1.0,
        "Dividends": 0.0,
        "Splits": 0.0,
    }
    return pd.DataFrame(data)


def test_check_asset_true():
    shareprice = make_shareprice(int(21 * 365.25) + 10)
    asset = AssetData(ticker="TEST", shareprice=shareprice)
    assert GroupOver20Years().checkAsset(asset)


def test_check_asset_false():
    # create only a few years of data so requirement is not met
    shareprice = make_shareprice(int(2 * 365.25))
    asset = AssetData(ticker="TEST", shareprice=shareprice)
    assert not GroupOver20Years().checkAsset(asset)
