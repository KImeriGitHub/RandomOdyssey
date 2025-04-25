import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars

class AssetDataService:
    def __init__(self):
        pass

    @staticmethod
    def defaultInstance(ticker: str = "", isin: str = "") -> AssetData:
        # Define column schemas for each DataFrame
        shareprice_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume', 'Dividends', 'Splits']
        quarterly_cols = [
            'fiscalDateEnding', 'reportedDate', 'reportedEPS', 'estimatedEPS', 'surprise',
            'surprisePercentage', 'reportTime', 'grossProfit', 'totalRevenue', 'ebit',
            'ebitda', 'totalAssets', 'totalCurrentLiabilities',
            'totalShareholderEquity', 'commonStockSharesOutstanding',
            'operatingCashflow'
        ]
        annual_cols = [
            'fiscalDateEnding', 'reportedEPS', 'grossProfit', 'totalRevenue', 'ebit', 'ebitda',
            'totalAssets', 'totalCurrentLiabilities', 'totalShareholderEquity',
            'operatingCashflow'
        ]

        # Create empty DataFrames with specified columns
        empty_shareprice = pd.DataFrame(columns=shareprice_cols)
        empty_quarterly = pd.DataFrame(columns=quarterly_cols)
        empty_annual = pd.DataFrame(columns=annual_cols)

        return AssetData(
            ticker=ticker,
            isin=isin,
            shareprice=empty_shareprice,
            about={},
            sector="",
            financials_quarterly=empty_quarterly,
            financials_annually=empty_annual
        )

    @staticmethod
    def to_dict(asset: AssetData) -> Dict:
        # Basic fields
        data = {
            "ticker": asset.ticker,
            "isin": asset.isin or "",
            "about": asset.about or {},
            "sector": asset.sector or "",
        }
        # Fallback empty instance for missing DataFrames
        empty = AssetDataService.defaultInstance(ticker=asset.ticker, isin=asset.isin)
        # Convert DataFrames to dict (default orient='dict')
        data['shareprice'] = asset.shareprice.to_dict() if isinstance(asset.shareprice, pd.DataFrame) else empty.shareprice.to_dict()
        data['financials_quarterly'] = asset.financials_quarterly.to_dict() if isinstance(asset.financials_quarterly, pd.DataFrame) else empty.financials_quarterly.to_dict()
        data['financials_annually'] = asset.financials_annually.to_dict() if isinstance(asset.financials_annually, pd.DataFrame) else empty.financials_annually.to_dict()
        return data
    
    @staticmethod
    def from_dict(assetdict: dict) -> AssetData:
        # Validate required fields
        ticker = assetdict.get("ticker")
        if not ticker:
            raise ValueError("Ticker symbol is required to construct AssetData.")

        # Create empty instance to get schema defaults
        instance = AssetDataService.defaultInstance(ticker=ticker, isin=assetdict.get("isin", ""))

        # Populate basic fields
        instance.about = assetdict.get("about", {}) or {}
        instance.sector = assetdict.get("sector", "") or ""

        # Helper to build DataFrame safely and match schema
        def build_df(data_dict, columns):
            df = pd.DataFrame(data_dict)
            return df.reindex(columns=columns)

        # Load shareprice
        sp_dict = assetdict.get("shareprice", {})
        instance.shareprice = build_df(sp_dict, instance.shareprice.columns)

        # Load financials quarterly
        fq_dict = assetdict.get("financials_quarterly", {})
        instance.financials_quarterly = build_df(fq_dict, instance.financials_quarterly.columns)

        # Load financials annually
        fa_dict = assetdict.get("financials_annually", {})
        instance.financials_annually = build_df(fa_dict, instance.financials_annually.columns)

        return instance
    
    @staticmethod
    def to_polars(ad: AssetData) -> AssetDataPolars:
        # Convert Pandas → Polars and cast date columns properly
        sp = pl.from_pandas(ad.shareprice).with_columns(
            pl.col("Date").cast(pl.Date)
        )
        fq = pl.from_pandas(ad.financials_quarterly).with_columns([
            pl.col("fiscalDateEnding").cast(pl.Date),
            pl.col("reportedDate").cast(pl.Date),
        ])
        fa = pl.from_pandas(ad.financials_annually).with_columns(
            pl.col("fiscalDateEnding").cast(pl.Date)
        )

        return AssetDataPolars(
            ticker=ad.ticker,
            isin=ad.isin,
            about=ad.about or {},
            sector=ad.sector or "",
            shareprice=sp,
            financials_quarterly=fq,
            financials_annually=fa,
        )
    
    @staticmethod
    def copy(ad: AssetData) -> AssetData:
        # Create a new instance of AssetData with the same attributes
        return AssetData(
            ticker=ad.ticker,
            isin=ad.isin,
            about=ad.about.copy() if ad.about else {},
            sector=ad.sector,
            shareprice=ad.shareprice.copy(),
            financials_quarterly=ad.financials_quarterly.copy(),
            financials_annually=ad.financials_annually.copy()
        )