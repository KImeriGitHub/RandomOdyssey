import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars

class AssetDataService:
    def __init__(self):
        pass

    @staticmethod
    def defaultInstance() -> AssetData:
        return AssetData(ticker = "", 
            isin = "", 
            shareprice = pd.DataFrame(None),
            adjClosePrice = pd.Series(None),
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
        data['adjClosePrice'] = asset.adjClosePrice.to_dict() if isinstance(asset.adjClosePrice, pd.Series) else pd.Series(None).to_dict()
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
        adjClosePriceDict = assetdict.get("adjClosePrice")
        volumeDict = assetdict.get("volume")
        dividendsDict = assetdict.get("dividends")
        splitsDict = assetdict.get("splits")
        revenueDict = assetdict.get("revenue")
        EBITDADict = assetdict.get("EBITDA")
        basicEPSDict = assetdict.get("basicEPS")

        defaultAD.ticker = assetdict["ticker"]
        defaultAD.isin = assetdict.get("isin") or ""
        defaultAD.shareprice = pd.DataFrame(sharepriceDict)
        defaultAD.adjClosePrice = pd.Series(adjClosePriceDict)
        defaultAD.volume = pd.Series(volumeDict)
        defaultAD.dividends = pd.Series(dividendsDict)
        defaultAD.splits = pd.Series(splitsDict)
        defaultAD.about = assetdict.get("about") or {}
        defaultAD.revenue = pd.Series(revenueDict)
        defaultAD.EBITDA = pd.Series(EBITDADict)
        defaultAD.basicEPS = pd.Series(basicEPSDict)

        return defaultAD
    
    @staticmethod
    def to_polars(ad: AssetData) -> AssetDataPolars:
        adpl = AssetDataPolars(ticker = ad.ticker, 
            isin = ad.isin, 
            shareprice = pl.DataFrame(None),
            adjClosePrice = pl.DataFrame(None),
            volume = pl.DataFrame(None),
            dividends = pl.DataFrame(None),
            splits = pl.DataFrame(None),
            about = ad.about,
            revenue = pl.DataFrame(None),
            EBITDA = pl.DataFrame(None),
            basicEPS = pl.DataFrame(None))
        
        # Convert and rename shareprice
        adpl.shareprice = pl.from_pandas(ad.shareprice.reset_index())
        adpl.shareprice = adpl.shareprice.rename({"index": "Date"})

        # Convert and rename shareprice
        adpl.adjClosePrice = pl.from_pandas(ad.adjClosePrice.reset_index())
        adpl.adjClosePrice = adpl.adjClosePrice.rename({"index": "Date", "0": "AdjClose"})

        # Convert and rename volume
        adpl.volume = pl.from_pandas(ad.volume.reset_index())
        adpl.volume = adpl.volume.rename({"index": "Date", "0": "Volume"})

        # Convert and rename dividends
        adpl.dividends = pl.from_pandas(ad.dividends.reset_index())
        adpl.dividends = adpl.dividends.rename({"index": "Date", "0": "Dividends"})

        # Convert and rename splits
        adpl.splits = pl.from_pandas(ad.splits.reset_index())
        adpl.splits = adpl.splits.rename({"index": "Date", "0": "Splits"})

        # Convert and rename revenue
        adpl.revenue = pl.from_pandas(ad.revenue.reset_index())
        adpl.revenue = adpl.revenue.rename({"index": "Date", "0": "Revenue"})

        # Convert and rename EBITDA
        adpl.EBITDA = pl.from_pandas(ad.EBITDA.reset_index())
        adpl.EBITDA = adpl.EBITDA.rename({"index": "Date", "0": "EBITDA"})

        # Convert and rename basicEPS
        adpl.basicEPS = pl.from_pandas(ad.basicEPS.reset_index())
        adpl.basicEPS = adpl.basicEPS.rename({"index": "Date", "0": "BasicEPS"})

        return adpl