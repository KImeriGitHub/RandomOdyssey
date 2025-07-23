import pandas as pd
import polars as pl
import re
from typing import Dict
from dataclasses import fields
from pandas.api.types import is_float_dtype
    
from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars

import logging
logger = logging.getLogger(__name__)

class AssetDataService:
    def __init__(self):
        pass

    @staticmethod
    def defaultInstance(ticker: str = "", isin: str = "") -> AssetData:
        # Define dtypes for shareprice
        shareprice_dtypes = {
            'Date': 'string',
            'Open': 'Float64',
            'High': 'Float64',
            'Low': 'Float64',
            'Close': 'Float64',
            'AdjClose': 'Float64',
            'Volume': 'Float64',
            'Dividends': 'Float64',
            'Splits': 'Float64'
        }
        # Define dtypes for quarterly financials
        quarterly_dtypes = {
            'fiscalDateEnding': 'string',
            'reportedDate': 'string',
            'reportedEPS': 'Float64',
            'estimatedEPS': 'Float64',
            'surprise': 'Float64',
            'surprisePercentage': 'Float64',
            'reportTime': 'string',
            'grossProfit': 'Float64',
            'totalRevenue': 'Float64',
            'ebit': 'Float64',
            'ebitda': 'Float64',
            'totalAssets': 'Float64',
            'totalCurrentLiabilities': 'Float64',
            'totalShareholderEquity': 'Float64',
            'commonStockSharesOutstanding': 'Float64',
            'operatingCashflow': 'Float64'
        }
        # Define dtypes for annual financials
        annual_dtypes = {
            'fiscalDateEnding': 'string',
            'reportedEPS': 'Float64',
            'grossProfit': 'Float64',
            'totalRevenue': 'Float64',
            'ebit': 'Float64',
            'ebitda': 'Float64',
            'totalAssets': 'Float64',
            'totalCurrentLiabilities': 'Float64',
            'totalShareholderEquity': 'Float64',
            'operatingCashflow': 'Float64'
        }

        # Create empty DataFrames with specified dtypes
        empty_shareprice = pd.DataFrame({col: pd.Series(dtype=dt)
            for col, dt in shareprice_dtypes.items()})
        empty_quarterly = pd.DataFrame({col: pd.Series(dtype=dt)
            for col, dt in quarterly_dtypes.items()})
        empty_annual = pd.DataFrame({col: pd.Series(dtype=dt)
            for col, dt in annual_dtypes.items()})

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

        # Load shareprice
        sp_dict = assetdict.get("shareprice", {})
        instance.shareprice = pd.DataFrame(sp_dict)

        # Load financials quarterly
        fq_dict = assetdict.get("financials_quarterly", {})
        instance.financials_quarterly = pd.DataFrame(fq_dict)

        # Load financials annually
        fa_dict = assetdict.get("financials_annually", {})
        instance.financials_annually = pd.DataFrame(fa_dict)

        return instance
    
    @staticmethod
    def to_polars(ad: AssetData) -> AssetDataPolars:
        # Convert Pandas → Polars and cast date columns properly
        sp = pl.from_pandas(ad.shareprice).with_columns(
            pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        ) if ad.shareprice is not None else None
        fq = pl.from_pandas(ad.financials_quarterly).with_columns([
            pl.col("fiscalDateEnding").str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
            pl.col("reportedDate").str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
        ]) if ad.financials_quarterly is not None else None
        fa = pl.from_pandas(ad.financials_annually).with_columns(
            pl.col("fiscalDateEnding").str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        ) if ad.financials_annually is not None else None

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

    @staticmethod
    def validate_asset_data(ad: AssetData):
        DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        VALID_REPORT_TIMES = {"pre-market", "post-market"}
        
        # 1. Field names
        dc_fields = {f.name for f in fields(AssetData)}
        inst_fields = set(ad.__dict__)
        if dc_fields != inst_fields:
            extra = inst_fields - dc_fields
            missing = dc_fields - inst_fields
            logger.error(f"VALIDATION ERROR: Field mismatch – extra: {extra}, missing: {missing}")

        # 2. Attribute types
        if not isinstance(ad.ticker, str):
            logger.error("VALIDATION ERROR: ticker must be str")
        if not isinstance(ad.isin, str):
            logger.error("VALIDATION ERROR: isin must be str")
        if not isinstance(ad.sector, str):
            logger.error("VALIDATION ERROR: sector must be str")
        if ad.about is not None and not isinstance(ad.about, dict):
            logger.error("VALIDATION ERROR: about must be dict or None")

        # Helper to check a date‐string column
        def check_date_col(df, col):
            if df[col].dtype != object:
                logger.error(f"VALIDATION ERROR: {col} must be dtype object (str), got {df[col].dtype}")
            bad = df[~df[col].astype(str).str.match(DATE_RE)]
            if not bad.empty:
                logger.error(f"VALIDATION ERROR: Column {col} has invalid dates:\n{bad[col].unique()}")

        # Helper to check float columns
        def check_float_col(df, col):
            if not is_float_dtype(df[col]):
                logger.error(f"VALIDATION ERROR: {col} must be float dtype, got {df[col].dtype}")

        # 3. shareprice
        if ad.shareprice is not None:
            exp = ['Date','Open','High','Low','Close','AdjClose','Volume','Dividends','Splits']
            if list(ad.shareprice.columns) != exp:
                logger.error(f"VALIDATION ERROR: shareprice.columns must be {exp}")
            check_date_col(ad.shareprice, 'Date')
            for c in exp[1:]:
                check_float_col(ad.shareprice, c)

        # 4. financials_quarterly
        if ad.financials_quarterly is not None:
            exp_q = [
                'fiscalDateEnding','reportedDate','reportedEPS','estimatedEPS','surprise',
                'surprisePercentage','reportTime','grossProfit','totalRevenue','ebit',
                'ebitda','totalAssets','totalCurrentLiabilities','totalShareholderEquity',
                'commonStockSharesOutstanding','operatingCashflow'
            ]
            if list(ad.financials_quarterly.columns) != exp_q:
                logger.error(f"VALIDATION ERROR: financials_quarterly.columns must be {exp_q}")
            check_date_col(ad.financials_quarterly, 'fiscalDateEnding')
            check_date_col(ad.financials_quarterly, 'reportedDate')
            # reportTime
            rt = ad.financials_quarterly['reportTime']
            if ad.financials_quarterly['reportTime'].dtype != object:
                logger.error(f"VALIDATION ERROR: 'reportTime' must be dtype object (str), got {ad.financials_quarterly['reportTime'].dtype}")
            if not rt.empty and (rt.dtype != object or not set(rt.unique()).issubset(VALID_REPORT_TIMES)):
                logger.error(f"VALIDATION ERROR: reportTime must be in {VALID_REPORT_TIMES}, got {rt.unique()}")
            # numeric
            for c in exp_q[2:] :
                if c not in {'reportTime'}:
                    check_float_col(ad.financials_quarterly, c)

        # 5. financials_annually
        if ad.financials_annually is not None:
            exp_a = [
                'fiscalDateEnding','reportedEPS','grossProfit','totalRevenue','ebit',
                'ebitda','totalAssets','totalCurrentLiabilities','totalShareholderEquity',
                'operatingCashflow'
            ]
            if list(ad.financials_annually.columns) != exp_a:
                logger.error(f"VALIDATION ERROR: financials_annually.columns must be {exp_a}")
            check_date_col(ad.financials_annually, 'fiscalDateEnding')
            for c in exp_a[1:]:
                check_float_col(ad.financials_annually, c)