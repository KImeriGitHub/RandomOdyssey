import pandas as pd
from dataclasses import dataclass
from typing import Dict

@dataclass
class AssetData:
    ticker: str
    isin: str = ""

    # [Open, High, Low, Close, Adj Close]
    shareprice: pd.DataFrame = None
    volume: pd.Series = None
    dividends: pd.Series = None
    splits: pd.Series = None

    adjClosePrice: pd.Series = None

    #General Information about the asset in dict format
    about: Dict = None

    #financials
    revenue: pd.Series = None
    EBITDA: pd.Series= None
    basicEPS: pd.Series = None