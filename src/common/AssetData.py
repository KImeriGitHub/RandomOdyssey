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
    dividends: pd.Series = None # Note: Must be per day and not in percentage
    splits: pd.Series = None

    # Adj Close has No NAN Values and maybe extra dates than Close price
    adjClosePrice: pd.Series = None

    #General Information about the asset in dict format
    about: Dict = None
    sector: str = ""

    #financials
    financials_quarterly: pd.DataFrame = None
    financials_annually: pd.DataFrame = None