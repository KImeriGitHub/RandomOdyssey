import pandas as pd
from dataclasses import dataclass

@dataclass
class AssetData:
    ticker: str
    isin: str
    shareprice: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame
    genInfo: dict               #todo: stock_info.info
    financials: pd.DataFrame    #todo: try getting quarterly ones. And drop most columns