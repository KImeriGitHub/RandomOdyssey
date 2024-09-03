import pandas as pd
from dataclasses import dataclass

@dataclass
class AssetData:
    ticker: str
    isin: str
    shareprice: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame
    about: dict               #todo: stock_info.info

    #financials
    revenue: pd.DataFrame
    EBITDA: pd.DataFrame
    basicEPS: pd.DataFrame