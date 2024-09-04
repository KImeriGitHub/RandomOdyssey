import pandas as pd
from dataclasses import dataclass

@dataclass
class AssetData:
    ticker: str
    isin: str

    # [Open, High, Low, Close]
    shareprice: pd.DataFrame
    volume: pd.Series
    dividends: pd.Series
    splits: pd.Series

    #General Information about the asset in dict format
    about: dict

    #financials
    revenue: pd.Series
    EBITDA: pd.Series
    basicEPS: pd.Series