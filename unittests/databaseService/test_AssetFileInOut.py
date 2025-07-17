import pandas as pd
from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetDataService import AssetDataService


def create_sample_asset() -> object:
    asset = AssetDataService.defaultInstance()
    asset.ticker = "TEST"

    # Minimal shareprice dataframe
    data = {
        "Date": pd.date_range("2020-01-01", periods=5, freq="D").astype(str),
        "Open": [1, 2, 3, 4, 5],
        "High": [1, 2, 3, 4, 5],
        "Low": [1, 2, 3, 4, 5],
        "Close": [1, 2, 3, 4, 5],
        "AdjClose": [1, 2, 3, 4, 5],
        "Volume": [10, 20, 30, 40, 50],
        "Dividends": [0, 0, 0, 0, 0],
        "Splits": [0, 0, 0, 0, 0],
    }
    asset.shareprice = pd.DataFrame(data)

    # Minimal financials dataframes
    q_cols = AssetDataService.defaultInstance().financials_quarterly.columns
    a_cols = AssetDataService.defaultInstance().financials_annually.columns
    asset.financials_quarterly = pd.DataFrame([{c: 0 for c in q_cols}])
    asset.financials_annually = pd.DataFrame([{c: 0 for c in a_cols}])
    return asset


def test_save_creates_file(tmp_path):
    file_io = AssetFileInOut(str(tmp_path))
    asset = AssetDataService.defaultInstance()
    asset.ticker = "TEST"

    file_io.saveToFile(asset)
    dest = tmp_path / "TEST.pkl"
    assert dest.exists()


def test_save_and_load_roundtrip(tmp_path):
    file_io = AssetFileInOut(str(tmp_path))
    asset = create_sample_asset()

    file_io.saveToFile(asset)
    loaded = file_io.loadFromFile("TEST")

    pd.testing.assert_frame_equal(asset.shareprice, loaded.shareprice)
    pd.testing.assert_frame_equal(asset.financials_quarterly, loaded.financials_quarterly)
    pd.testing.assert_frame_equal(asset.financials_annually, loaded.financials_annually)
    assert asset.ticker == loaded.ticker
    assert asset.isin == loaded.isin
