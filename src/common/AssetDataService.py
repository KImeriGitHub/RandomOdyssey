import pandas as pd
from typing import Dict

from src.common.AssetData import AssetData

class AssetDataService:
    def __init__(self):
        pass

    @staticmethod
    def defaultInstance() -> AssetData:
        return AssetData(ticker = "", 
            isin = "", 
            shareprice = pd.DataFrame(None),
            volume = pd.Series(None),
            dividends = pd.Series(None),
            splits = pd.Series(None),
            about = {},
            revenue = pd.Series(None),
            EBITDA = pd.Series(None),
            basicEPS = pd.Series(None))

    @staticmethod
    def to_dict(asset: AssetData) -> Dict:
        # Dictionary of basic fields with default values
        data = {
            "ticker": asset.ticker,
            "isin": asset.isin or "",
            "about": asset.about or {},
        }

        data['shareprice'] = asset.shareprice.to_dict() if isinstance(asset.shareprice, pd.DataFrame) else pd.DataFrame(None).to_dict()
        data['volume'] = asset.volume.to_dict() if isinstance(asset.volume, pd.Series) else pd.Series(None).to_dict()
        data['dividends'] = asset.dividends.to_dict() if isinstance(asset.dividends, pd.Series) else pd.Series(None).to_dict()
        data['splits'] = asset.splits.to_dict() if isinstance(asset.splits, pd.Series) else pd.Series(None).to_dict()
        data['revenue'] = asset.revenue.to_dict() if isinstance(asset.revenue, pd.Series) else pd.Series(None).to_dict()
        data['EBITDA'] = asset.EBITDA.to_dict() if isinstance(asset.EBITDA, pd.Series) else pd.Series(None).to_dict()
        data['basicEPS'] = asset.basicEPS.to_dict() if isinstance(asset.basicEPS, pd.Series) else pd.Series(None).to_dict()

        return data
    
    @staticmethod
    def from_dict(assetdict: dict) -> AssetData:
        defaultAD: AssetData = AssetDataService.defaultInstance()

        if assetdict.get("ticker") is None:
            raise ValueError("Ticker symbol could not be loaded. (From from_dict in AssetDataService)")
        
        sharepriceDict = assetdict.get("shareprice")
        volumeDict = assetdict.get("volume")
        dividendsDict = assetdict.get("dividends")
        splitsDict = assetdict.get("splits")
        revenueDict = assetdict.get("revenue")
        EBITDADict = assetdict.get("EBITDA")
        basicEPSDict = assetdict.get("basicEPS")

        defaultAD.ticker = assetdict["ticker"]
        defaultAD.isin = assetdict.get("isin") or ""
        defaultAD.shareprice = pd.DataFrame(sharepriceDict)
        defaultAD.volume = pd.Series(volumeDict)
        defaultAD.dividends = pd.Series(dividendsDict)
        defaultAD.splits = pd.Series(splitsDict)
        defaultAD.about = assetdict.get("about") or {}
        defaultAD.revenue = pd.Series(revenueDict)
        defaultAD.EBITDA = pd.Series(EBITDADict)
        defaultAD.basicEPS = pd.Series(basicEPSDict)

        return defaultAD