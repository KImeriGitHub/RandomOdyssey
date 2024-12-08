import pandas as pd
import polars as pl
import pyarrow
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
            sector = "",
            financials_quarterly = pd.DataFrame(None),
            financials_annually = pd.DataFrame(None),
        )

    @staticmethod
    def to_dict(asset: AssetData) -> Dict:
        # Dictionary of basic fields with default values
        data = {
            "ticker": asset.ticker,
            "isin": asset.isin or "",
            "about": asset.about or {},
            "sector": asset.sector or "",
        }

        data['shareprice'] = asset.shareprice.to_dict() if isinstance(asset.shareprice, pd.DataFrame) else pd.DataFrame(None).to_dict()
        data['adjClosePrice'] = asset.adjClosePrice.to_dict() if isinstance(asset.adjClosePrice, pd.Series) else pd.Series(None).to_dict()
        data['volume'] = asset.volume.to_dict() if isinstance(asset.volume, pd.Series) else pd.Series(None).to_dict()
        data['dividends'] = asset.dividends.to_dict() if isinstance(asset.dividends, pd.Series) else pd.Series(None).to_dict()
        data['splits'] = asset.splits.to_dict() if isinstance(asset.splits, pd.Series) else pd.Series(None).to_dict()
        data['financials_quarterly'] = asset.financials_quarterly.to_dict() if isinstance(asset.financials_quarterly, pd.DataFrame) else pd.DataFrame(None).to_dict()
        data['financials_annually'] = asset.financials_annually.to_dict() if isinstance(asset.financials_annually, pd.DataFrame) else pd.DataFrame(None).to_dict()

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
        financialsQuarDict = assetdict.get("financials_quarterly")
        financialsAnDict = assetdict.get("financials_annually")

        defaultAD.ticker = assetdict["ticker"]
        defaultAD.isin = assetdict.get("isin") or ""
        defaultAD.shareprice = pd.DataFrame(sharepriceDict)
        defaultAD.adjClosePrice = pd.Series(adjClosePriceDict)
        defaultAD.volume = pd.Series(volumeDict)
        defaultAD.dividends = pd.Series(dividendsDict)
        defaultAD.splits = pd.Series(splitsDict)
        defaultAD.about = assetdict.get("about") or {}
        defaultAD.sector = assetdict.get("sector") or {}
        defaultAD.financials_quarterly = pd.DataFrame(financialsQuarDict)
        defaultAD.financials_annually = pd.DataFrame(financialsAnDict)

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
            sector = ad.sector,
            financials_quarterly = pl.DataFrame(None),
            financials_annually = pl.DataFrame(None),
            )
        
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
        adpl.financials_quarterly = pl.from_pandas(ad.financials_quarterly)

        # Convert and rename EBITDA
        adpl.financials_annually = pl.from_pandas(ad.financials_annually)

        return adpl