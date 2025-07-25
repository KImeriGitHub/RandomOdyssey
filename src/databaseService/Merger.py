import pandas as pd
import polars as pl
from datetime import datetime as dt
import datetime
from src.common.AssetData import AssetData

from src.databaseService.CleanData import CleanData

import logging

logger = logging.getLogger(__name__)

class Merger():
    def __init__(self, assetData: AssetData):
        # No longer storing data in the constructor
        self.asset = assetData
    
    def merge_shareprice(self, mergingshareprice: pd.DataFrame) -> None:
        """
        Merges share price data into the asset's shareprice DataFrame.
        """
        if mergingshareprice.empty: 
            raise ValueError("Mergingshareprice DataFrame is empty.")
        
        shareprice_dtypes_dict = {
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
        
        ### PREPARE MERGING SHAREPRICE DATA
        fullSharePrice = mergingshareprice.copy()
        fullSharePrice["Date"] = fullSharePrice["Date"].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        
        ### IF NO SHAREPRICE DATA IN DB, JUST ASSIGN
        if len(self.asset.shareprice) == 0:
            fullSharePrice["Date"] = fullSharePrice["Date"].apply(lambda ts: str(ts))
            self.asset.shareprice = fullSharePrice.astype(shareprice_dtypes_dict, errors='raise')
            logger.info(f"  No existing shareprice data for ticker {self.asset.ticker}.")
            return
            
        ### PREPARE TO MERGE SHAREPRICE DATA
        full = fullSharePrice.copy()
        existing = self.asset.shareprice.copy()
        existing['Date'] = existing['Date'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        
        existing_pl = pl.from_pandas(existing).sort('Date')
        full_pl = pl.from_pandas(full).sort('Date')
        
        merged_pl = existing_pl.join(
            full_pl,
            on="Date",
            how="full",
            coalesce=True,
            suffix="_new"
        )
        merged_pl = merged_pl.sort("Date")
        
        # Update null Entries
        merged_pl = merged_pl.with_columns(
            pl.when(pl.col("Open").is_null()).then(pl.col("Open_new")).otherwise(pl.col("Open")).alias("Open"),
            pl.when(pl.col("High").is_null()).then(pl.col("High_new")).otherwise(pl.col("High")).alias("High"),
            pl.when(pl.col("Low").is_null()).then(pl.col("Low_new")).otherwise(pl.col("Low")).alias("Low"),
            pl.when(pl.col("Close").is_null()).then(pl.col("Close_new")).otherwise(pl.col("Close")).alias("Close"),
            pl.when(pl.col("Volume").is_null()).then(pl.col("Volume_new")).otherwise(pl.col("Volume")).alias("Volume"),
            pl.when(pl.col("Dividends").is_null()).then(pl.col("Dividends_new")).otherwise(pl.col("Dividends")).alias("Dividends"),
            pl.when(pl.col("Splits").is_null()).then(pl.col("Splits_new")).otherwise(pl.col("Splits")).alias("Splits")
        )
        
        # Update AdjClose
        # anchor ratio from known rows
        merged_pl = (
            merged_pl
            .sort("Date")
            # 1) anchor AdjClose where present
            .with_columns(
                pl.col("AdjClose_new").alias("AdjClose")
            )
            .with_columns(
                (pl.col("AdjClose") / pl.col("Close")).alias("ratio_known")
            )
            # 2) per‑day total‑return event factor
            .with_columns(
                (
                    1 / pl.coalesce([pl.col("Splits"), pl.lit(1.0)])
                    * ((pl.col("Close") - pl.coalesce([pl.col("Dividends"), pl.lit(0.0)])) / pl.col("Close"))
                ).alias("ev_fac")
            )
            # 3) reverse cumulative product of future events
            .with_columns(
                pl.col("ev_fac").reverse().cum_prod().reverse().alias("adj_fac")
            )
            # 4) scale to match known AdjClose ratios
            .with_columns(
                pl.when(pl.col("ratio_known").is_not_null())
                .then(pl.col("ratio_known") / pl.col("adj_fac"))
                .otherwise(None)
                .alias("scale")
            )
            .with_columns(
                pl.col("scale")
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .alias("scale")
            )
            # 5) fill the remaining gaps
            .with_columns(
                pl.when(pl.col("AdjClose").is_null())
                .then(pl.col("Close") * pl.col("adj_fac") * pl.col("scale"))
                .otherwise(pl.col("AdjClose"))
                .alias("AdjClose")
            )
            # 6) clean up
            .drop(["AdjClose_new", "ratio_known", "ev_fac", "adj_fac", "scale"])
        )
        
        # New columns with ratios (for logging)
        merged_pl = merged_pl.with_columns(
            (pl.col("Open_new") / pl.col("Open")).alias("Open_ratio"),
            (pl.col("High_new") / pl.col("High")).alias("High_ratio"),
            (pl.col("Low_new") / pl.col("Low")).alias("Low_ratio"),
            (pl.col("Close_new") / pl.col("Close")).alias("Close_ratio"),
            ((pl.col("Volume_new") - pl.col("Volume"))/100.0).alias("Volume_diff"),
            (pl.col("Dividends_new") - pl.col("Dividends")).alias("Dividends_diff"),
            (pl.col("Splits_new") / pl.col("Splits")).alias("Splits_ratio")
        )
        
        # Adopt new values
        merged_pl = merged_pl.with_columns(
            pl.when(pl.col("Open").is_not_null() & pl.col("Open_new").is_not_null()).then(pl.col("Open_new")).otherwise(pl.col("Open")).alias("Open"),
            pl.when(pl.col("High").is_not_null() & pl.col("High_new").is_not_null()).then(pl.col("High_new")).otherwise(pl.col("High")).alias("High"),
            pl.when(pl.col("Low").is_not_null() & pl.col("Low_new").is_not_null()).then(pl.col("Low_new")).otherwise(pl.col("Low")).alias("Low"),
            pl.when(pl.col("Close").is_not_null() & pl.col("Close_new").is_not_null()).then(pl.col("Close_new")).otherwise(pl.col("Close")).alias("Close"),
            pl.when(pl.col("Volume").is_not_null() & pl.col("Volume_new").is_not_null()).then(pl.col("Volume_new")).otherwise(pl.col("Volume")).alias("Volume"),
            pl.when(pl.col("Dividends").is_not_null() & pl.col("Dividends_new").is_not_null()).then(pl.col("Dividends_new")).otherwise(pl.col("Dividends")).alias("Dividends"),
            pl.when(pl.col("Splits").is_not_null() & pl.col("Splits_new").is_not_null()).then(pl.col("Splits_new")).otherwise(pl.col("Splits")).alias("Splits")
        )
        
        # Logging of critical changes
        counts_df = merged_pl.select([
            ((pl.col(c) > 1.01) | (pl.col(c) < 0.99)).sum().alias(c)
            for c in ["Open_ratio", "High_ratio", "Low_ratio", "Close_ratio", "Splits_ratio"]
        ] + [
            ((pl.col(c) > 0.01) | (pl.col(c) < -0.01)).sum().alias(c)
            for c in ["Volume_diff", "Dividends_diff"]
        ])
        counts = counts_df.to_dicts()[0]
        for col, cnt in counts.items():
            if cnt > 0:
                logger.info(f"  {col}: {cnt} values outside +-1%")
        
        ### ASSIGN MERGED SHAREPRICE DATA
        merged_pl = merged_pl.select(["Date",'Open','High','Low','Close','AdjClose','Volume','Dividends','Splits'])
        merged_pd = merged_pl.to_pandas()
        merged_pd['Date'] = merged_pd['Date'].apply(lambda ts: str(ts.date()))
        merged_pd = merged_pd.astype(shareprice_dtypes_dict, errors='raise')
        self.asset.shareprice = merged_pd
        
        # Further logging
        new_dates = set(full_pl["Date"].to_list()) - set(existing_pl["Date"].to_list())
        logger.info(f"  Added {len(new_dates)} new rows to shareprice data of ticker {self.asset.ticker}.")
            
    def merge_financials(self, fin_ann: pd.DataFrame, fin_quar: pd.DataFrame) -> None:
        # Casting Annual Financials
        full_ann = fin_ann.copy()
        full_ann['fiscalDateEnding'] = full_ann['fiscalDateEnding'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        
        # Casting Quarterly Financials
        full_quar = fin_quar.copy()
        full_quar['fiscalDateEnding'] = full_quar['fiscalDateEnding'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        
        # Casting existing Financials
        existing_ann = self.asset.financials_annually.copy()
        existing_ann['fiscalDateEnding'] = existing_ann['fiscalDateEnding'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        existing_quar = self.asset.financials_quarterly.copy()
        existing_quar['fiscalDateEnding'] = existing_quar['fiscalDateEnding'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())

        old_ann  = existing_ann.copy()
        old_quar = existing_quar.copy()

        # Merging
        existing_ann = (
            full_ann.set_index('fiscalDateEnding')
            .combine_first(existing_ann.set_index('fiscalDateEnding'))
            .reset_index()
        ) if not existing_ann.empty else full_ann

        existing_quar = (
            full_quar.set_index('fiscalDateEnding')
            .combine_first(existing_quar.set_index('fiscalDateEnding'))
            .reset_index()
        ) if not existing_quar.empty else full_quar

        existing_ann = existing_ann.sort_values('fiscalDateEnding').reset_index(drop=True)
        existing_quar = existing_quar.sort_values('fiscalDateEnding').reset_index(drop=True)

        # Clean incongruencies
        existing_ann = CleanData.financial_fiscalDateIncongruence(existing_ann, daysDiscrep = 60)
        existing_quar = CleanData.financial_fiscalDateIncongruence(existing_quar, daysDiscrep = 15)

        # BEFORE recasting to string -> log
        self._log_fin_changes("annual",    old_ann,  existing_ann)
        self._log_fin_changes("quarterly", old_quar, existing_quar)

        # Recasting
        existing_ann['fiscalDateEnding']  = existing_ann['fiscalDateEnding'].apply(lambda ts: str(ts))
        existing_quar['fiscalDateEnding'] = existing_quar['fiscalDateEnding'].apply(lambda ts: str(ts))
        
        existing_ann = existing_ann.astype({
            'fiscalDateEnding'        : 'string',
            'reportedEPS'             : 'Float64',
            'grossProfit'             : 'Float64',
            'totalRevenue'            : 'Float64',
            'ebit'                    : 'Float64',
            'ebitda'                  : 'Float64',
            'totalAssets'             : 'Float64',
            'totalCurrentLiabilities' : 'Float64',
            'totalShareholderEquity'  : 'Float64',
            'operatingCashflow'       : 'Float64'
        }, errors='raise')
        existing_quar = existing_quar.astype({
            'fiscalDateEnding'            : 'string',
            'reportedDate'                : 'string',
            'reportedEPS'                 : 'Float64',
            'estimatedEPS'                : 'Float64',
            'surprise'                    : 'Float64',
            'surprisePercentage'          : 'Float64',
            'reportTime'                  : 'string',
            'grossProfit'                 : 'Float64',
            'totalRevenue'                : 'Float64',
            'ebit'                        : 'Float64',
            'ebitda'                      : 'Float64',
            'totalAssets'                 : 'Float64',
            'totalCurrentLiabilities'     : 'Float64',
            'totalShareholderEquity'      : 'Float64',
            'commonStockSharesOutstanding': 'Float64',
            'operatingCashflow'           : 'Float64'
        }, errors='raise')
        
        self.asset.financials_annually = existing_ann
        self.asset.financials_quarterly = existing_quar
        
    def _log_fin_changes(self, kind: str, old_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
        if old_df is None or old_df.empty:
            logger.info("%s: first load (%d rows)", kind, len(new_df))
            return

        key = "fiscalDateEnding"

        # 1) new lines
        new_dates = new_df.loc[~new_df[key].isin(old_df[key]), key].tolist()
        if len(new_dates) == 0:
            logger.info("%s: no new rows", kind)
            return
        logger.info("%s: +%d new rows %s", kind, len(new_dates), new_dates)

        # 2) differences vs previous periods
        old_c = old_df[old_df[key].isin(new_df[key])].sort_values(key).reset_index(drop=True)
        new_c = new_df[new_df[key].isin(old_df[key])].sort_values(key).reset_index(drop=True)
        diff = new_c.compare(old_c, keep_equal=False)
        logger.info("%s diffs: (%d rows, %d cells)",
                    kind,
                    diff.index.nunique(),
                    diff.notna().sum().sum()) if not diff.empty else None