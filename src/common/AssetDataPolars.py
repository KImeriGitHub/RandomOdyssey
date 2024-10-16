import polars as pl
from dataclasses import dataclass
from typing import Dict

@dataclass
class AssetDataPolars:
    ticker: str
    isin: str = ""

    # [Open, High, Low, Close]
    shareprice: pl.DataFrame = None
    volume: pl.DataFrame = None
    dividends: pl.DataFrame = None
    splits: pl.DataFrame = None

    #General Information about the asset in dict format
    about: Dict = None

    #financials
    revenue: pl.DataFrame = None
    EBITDA: pl.DataFrame= None
    basicEPS: pl.DataFrame = None