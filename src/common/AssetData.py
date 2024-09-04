import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class AssetData:
    ticker: str
    isin: str = ""

    # [Open, High, Low, Close]
    shareprice: pd.DataFrame = None
    volume: pd.Series = None
    dividends: pd.Series = None
    splits: pd.Series = None

    #General Information about the asset in dict format
    about: Dict = None

    #financials
    revenue: pd.Series = None
    EBITDA: pd.Series= None
    basicEPS: pd.Series = None

    def to_dict(self):
        # Convert pandas objects to dict if they are not None
        data = asdict(self)  # This gets the basic dict representation
        if isinstance(self.shareprice, pd.DataFrame):
            data['shareprice'] = self.shareprice.to_dict()
        if isinstance(self.volume, pd.Series):
            data['volume'] = self.volume.to_dict()
        if isinstance(self.dividends, pd.Series):
            data['dividends'] = self.dividends.to_dict()
        if isinstance(self.splits, pd.Series):
            data['splits'] = self.splits.to_dict()
        if isinstance(self.revenue, pd.Series):
            data['revenue'] = self.revenue.to_dict()
        if isinstance(self.EBITDA, pd.Series):
            data['EBITDA'] = self.EBITDA.to_dict()
        if isinstance(self.basicEPS, pd.Series):
            data['basicEPS'] = self.basicEPS.to_dict()

        return data
# Default assetData
"""
AssetData(ticker = "", 
        isin = "", 
        shareprice = None,
        volume = None,
        dividends = None,
        splits = None,
        about = None,
        revenue = None,
        EBITDA = None,
        basicEPS = None)
"""