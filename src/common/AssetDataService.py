import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict

from src.common.AssetData import AssetData

@dataclass
class AssetDataService:
    def __init__(self):
        pass

    @staticmethod
    def defaultInstance() -> AssetData:
        return AssetData(ticker = "", 
            isin = "", 
            shareprice = None,
            volume = None,
            dividends = None,
            splits = None,
            about = None,
            revenue = None,
            EBITDA = None,
            basicEPS = None)

    @staticmethod
    def to_dict(asset: AssetData) -> Dict:
    # Dictionary of basic fields with default values
        data = {
            "ticker": asset.ticker,
            "isin": asset.isin or "",
            "about": asset.about or {},
        }
        
        if isinstance(asset.shareprice, pd.DataFrame):
            data['shareprice'] = asset.shareprice.to_dict()
        if isinstance(asset.volume, pd.Series):
            data['volume'] = asset.volume.to_dict()
        if isinstance(asset.dividends, pd.Series):
            data['dividends'] = asset.dividends.to_dict()
        if isinstance(asset.splits, pd.Series):
            data['splits'] = asset.splits.to_dict()
        if isinstance(asset.revenue, pd.Series):
            data['revenue'] = asset.revenue.to_dict()
        if isinstance(asset.EBITDA, pd.Series):
            data['EBITDA'] = asset.EBITDA.to_dict()
        if isinstance(asset.basicEPS, pd.Series):
            data['basicEPS'] = asset.basicEPS.to_dict()

        return data
    
    @staticmethod
    def from_dict(assetdict: dict) -> AssetData:
        defaultAD: AssetData = AssetDataService.defaultInstance()

        sharepriceDict = assetdict["shareprice"]
        volumeDict = assetdict["volume"]
        dividendsDict = assetdict["dividends"]
        splitsDict = assetdict["splits"]
        revenueDict = assetdict["revenue"]
        EBITDADict = assetdict["EBITDA"]
        basicEPSDict = assetdict["basicEPS"]

        if assetdict["ticker"] is None:
            raise ValueError("Ticker symbol could not be loaded. (From from_dict in AssetDataService)")
        
        defaultAD.ticker = assetdict["ticker"]
        defaultAD.isin = assetdict.get("isin") or ""
        defaultAD.shareprice = pd.DataFrame(sharepriceDict)
        defaultAD.volume = pd.Series(volumeDict)
        defaultAD.dividends = pd.Series(dividendsDict)
        defaultAD.splits = pd.Series(splitsDict)
        defaultAD.about = assetdict.get("about") or ""
        defaultAD.revenue = pd.Series(revenueDict)
        defaultAD.EBITDA = pd.Series(EBITDADict)
        defaultAD.basicEPS = pd.Series(basicEPSDict)

        return defaultAD