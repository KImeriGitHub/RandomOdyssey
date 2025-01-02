import polars as pl
from dataclasses import dataclass
from typing import Dict

@dataclass
class AssetDataPolars:
    ticker: str
    isin: str = ""

    # [Date, Open, High, Low, Close, Adj Close]
    shareprice: pl.DataFrame = None
    volume: pl.DataFrame = None
    dividends: pl.DataFrame = None  # Note: Must be per day and not in percentage
    splits: pl.DataFrame = None

    # Adj Close has No NAN Values and maybe extra dates than Close price
    adjClosePrice: pl.DataFrame = None

    #General Information about the asset in dict format
    about: Dict = None
    sector: str = ""

    #financials
    financials_quarterly: pl.DataFrame = None
    financials_annually: pl.DataFrame = None