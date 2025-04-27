import pandas as pd
from datetime import datetime as dt
from typing import Dict, Tuple, List
from src.common.AssetData import AssetData
import logging

logger = logging.getLogger(__name__)

# For Alpha Vantage
class Merger_AV():
    def __init__(self, assetData: AssetData):
        # No longer storing data in the constructor
        self.asset = assetData
    
    def merge_shareprice(self, fullSharePrice: pd.DataFrame) -> None:
        """
        Merges share price data into the asset's shareprice DataFrame.
        """
        fullSharePrice.reset_index(inplace=True)
        fullSharePrice = fullSharePrice.iloc[::-1] #flip upside down
        fullSharePrice.rename(columns={
            'date': 'Date',
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'AdjClose',
            '6. volume': 'Volume',
            '7. dividend amount': 'Dividends',
            '8. split coefficient': 'Splits'
        }, inplace=True)
        cols = ['Open','High','Low','Close','AdjClose','Volume','Dividends','Splits']
        
        # Drop missing dates
        if pd.isnull(fullSharePrice['Date']).any() or fullSharePrice['Date'].isna().any():
            logger.info(f"  Shareprice has empty dates or NaT Dates.")
            fullSharePrice = fullSharePrice[pd.notnull(fullSharePrice['Date'])] # remove empty Date columns
            fullSharePrice = fullSharePrice[fullSharePrice['Date'].notna()] # remove NaT Date columns
        fullSharePrice['Date'] = fullSharePrice['Date'].apply(lambda ts: ts.date())
        
        # Drop missing values
        if pd.isnull(fullSharePrice[cols]).any().any() or fullSharePrice[cols].isna().any().any():
            logger.info(f"  Shareprice has empty Number values or NaN Number values.")
            for col in cols:
                fullSharePrice = fullSharePrice[pd.notnull(fullSharePrice[col])]
                fullSharePrice = fullSharePrice[fullSharePrice[col].notna()]
            
        
        full = fullSharePrice.copy()
        existing = self.asset.shareprice.copy()
        existing['Date'] = existing['Date'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        
        new_rows = []
        diffs = []

        for _, new in full.iterrows():
            date = new['Date']
            mask = existing['Date'] == date
            if not mask.any():
                new_rows.append(new)
            else:
                old = existing.loc[mask].iloc[0]
                for c in cols:
                    o, n = old[c], new[c]
                    if o == 0:
                        if n != 0:
                            diffs.append(f"{date} {c}: old=0 -> new={n}")
                    elif abs(n - o) / abs(o) > 0.01:
                        pct = (n - o) / o * 100
                        diffs.append(f"{date} {c}: {pct:.2f}% (old={o}, new={n})")

        # append new rows
        if new_rows:
            appended = pd.DataFrame(new_rows)
            concat = pd.concat([existing, appended], ignore_index=True) if not existing.empty else appended
            concat = concat.sort_values('Date').reset_index(drop=True)
            concat['Date'] = concat['Date'].apply(lambda ts: str(ts))
            self.asset.shareprice = concat
            
            logger.info(f"  Added {len(new_rows)} new rows to shareprice data of ticker {self.asset.ticker}.")
        else:
            logger.info(f"  No new rows added to shareprice data of ticker {self.asset.ticker}.")
            
        # print summary
        if diffs:
            logger.info(f"  Changes to old data amount to >1%:\n" + "\n".join(diffs))
            
    def merge_financials(self, fin_ann: pd.DataFrame, fin_quar: pd.DataFrame) -> None:
        # Assert
        assert pd.api.types.is_datetime64_any_dtype(fin_ann['fiscalDateEnding']), (
            "fin_ann.fiscalDateEnding must be datetime64[ns], "
            f"got {fin_ann['fiscalDateEnding'].dtype}"
        )
        assert pd.api.types.is_datetime64_any_dtype(fin_quar['fiscalDateEnding']), (
            "fin_quar.fiscalDateEnding must be datetime64[ns], "
            f"got {fin_quar['fiscalDateEnding'].dtype}"
        )
        assert pd.api.types.is_datetime64_any_dtype(fin_quar['reportedDate']), (
            "fin_quar.reportedDate must be datetime64[ns], "
            f"got {fin_quar['reportedDate'].dtype}"
        )
        # Casting Annual Financials
        full_ann = fin_ann.copy()
        if pd.isnull(full_ann['fiscalDateEnding']).any():   # for empty dates
            logger.info(f"  Annual Financials has empty dates.")
            full_ann = full_ann[pd.notnull(full_ann['fiscalDateEnding'])]
        if full_ann["fiscalDateEnding"].isna().any():        # for NaT dates
            logger.info(f"  Annual Financials has NaT dates.")
            full_ann = full_ann[full_ann["fiscalDateEnding"].notna()]
        full_ann['fiscalDateEnding'] = full_ann['fiscalDateEnding'].apply(lambda ts: ts.date())
        
        # Casting Quarterly Financials
        full_quar = fin_quar.copy()
        if pd.isnull(full_quar['fiscalDateEnding']).any():
            logger.info(f"  Quarterly Financials has empty fiscal Dates.")
            full_quar = full_quar[pd.notnull(full_quar['fiscalDateEnding'])]
        if full_quar["fiscalDateEnding"].isna().any():
            logger.info(f"  Quarterly Financials has NaT fiscal Dates.")
            full_quar = full_quar[full_quar["fiscalDateEnding"].notna()]
        if pd.isnull(full_quar['reportedDate']).any():
            logger.info(f"  Quarterly Financials has empty reported Dates.")
            full_quar = full_quar[pd.notnull(full_quar['reportedDate'])]
        if full_quar["reportedDate"].isna().any():
            logger.info(f"  Quarterly Financials has NaT reported Dates.")
            full_quar = full_quar[full_quar["reportedDate"].notna()]
        full_quar['fiscalDateEnding'] = full_quar['fiscalDateEnding'].apply(lambda ts: ts.date())
        full_quar['reportedDate'] = full_quar['reportedDate'].apply(
            lambda x: x.date().__str__()
        )
        
        # Casting existing Financials
        existing_ann = self.asset.financials_annually.copy()
        existing_ann['fiscalDateEnding'] = existing_ann['fiscalDateEnding'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        existing_quar = self.asset.financials_quarterly.copy()
        existing_quar['fiscalDateEnding'] = existing_quar['fiscalDateEnding'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        
        cols_ann = [
            'fiscalDateEnding','reportedEPS','grossProfit','totalRevenue','ebit','ebitda',
            'totalAssets','totalCurrentLiabilities','totalShareholderEquity',
            'operatingCashflow'
        ]
        cols_quar = [
            'fiscalDateEnding','reportedDate','reportedEPS','estimatedEPS','surprise',
            'surprisePercentage','reportTime','grossProfit','totalRevenue','ebit',
            'ebitda','totalAssets','totalCurrentLiabilities','totalShareholderEquity',
            'commonStockSharesOutstanding','operatingCashflow'
        ]
        full_ann = full_ann[cols_ann]
        full_quar = full_quar[cols_quar]

        cn_updated_ann = 0
        for _, new in full_ann.iterrows():
            date: dt.date = new['fiscalDateEnding']
            mask = existing_ann['fiscalDateEnding'] == date
            if mask.any():
                ex_idx = existing_ann[mask].index[0]
                for col in existing_ann.columns.drop('fiscalDateEnding'):
                    if pd.isna(existing_ann.at[ex_idx, col]):
                        existing_ann.at[ex_idx, col] = new[col]
                        cn_updated_ann += 1
            else:
                # 2) A has not that date: check if any in same year
                year_mask = existing_ann['fiscalDateEnding'].apply(lambda d: d.year) == date.year
                if not year_mask.any():
                    # 2a) no entries in that year → append B’s row
                    if existing_ann.empty:
                        existing_ann = pd.DataFrame([new], columns=existing_ann.columns)
                    elif pd.DataFrame([new], columns=existing_ann.columns).empty:
                        pass
                    else:
                        existing_ann = pd.concat((existing_ann, pd.DataFrame([new], columns=existing_ann.columns)), ignore_index=True)
                    cn_updated_ann += 1
                else:
                    # 2b) some entry in that year → log and skip
                    logger.info(f"  DB annual fiscal date differs from new date. Year {date.year}.")
        
        today = dt.now().date()
        cn_updated_quar = 0
        for _, new in full_quar.iterrows():
            date = new['fiscalDateEnding']
            mask = existing_quar['fiscalDateEnding'] == date
            
            # 1) exact‐date match -> fill in any missing fields
            if mask.any():
                for col in existing_quar.columns.drop('fiscalDateEnding'):
                    if pd.isna(existing_quar.loc[mask, col].iloc[0]):
                        existing_quar.loc[mask, col] = new[col]
                        cn_updated_quar += 1
            else:
                age = (today - date).days
                
                # 2.1) newly reported -> append as new row
                if 0 <= age <= 60:
                    if existing_quar.empty:
                        existing_quar = pd.DataFrame([new], columns=existing_quar.columns)
                    elif pd.DataFrame([new]).empty:
                        pass
                    else:
                        existing_quar = pd.concat([existing_quar, pd.DataFrame([new], columns=existing_quar.columns)], ignore_index=True)
                    logger.info(f"  New quarterly fiscal statement in last month.")
                    cn_updated_quar += 1
                    
                else:
                    # 2.2) determine quarter and look for any record in same Q/Y
                    q = (date.month - 1) // 3 + 1
                    mask_in_q = (existing_quar['fiscalDateEnding'].apply(lambda x: x.year) == date.year) & \
                           (existing_quar['fiscalDateEnding'].apply(lambda x: (x.month - 1) // 3 + 1) == q)
                    
                    # 3.1) no record for this quarter -> append
                    if not mask_in_q.any():
                        if existing_quar.empty:
                            existing_quar = pd.DataFrame([new], columns=existing_quar.columns)
                        elif pd.DataFrame([new]).empty:
                            pass
                        else:
                            existing_quar = pd.concat([existing_quar, pd.DataFrame([new], columns=existing_quar.columns)], ignore_index=True)
                        cn_updated_quar += 1
                    else:
                        # 3.2) same quarter but different date -> update missing fields and bump date
                        for col in existing_quar.columns.drop('fiscalDateEnding'):
                            if pd.isna(existing_quar.loc[mask_in_q, col].iloc[0]):
                                existing_quar.loc[mask_in_q, col] = new[col]
                                cn_updated_quar += 1
                        existingDate = existing_quar.loc[mask_in_q, 'fiscalDateEnding'].iloc[0]
                        logger.info(f"  DB quarterly fiscal date differs from new date.")
                        logger.info(f"    Existing: Year {existingDate.year} Month {existingDate.month}. New: Year {date.year} Month {date.month}.")            
        
        #Recasting
        existing_ann = existing_ann.sort_values('fiscalDateEnding').reset_index(drop=True)
        existing_quar = existing_quar.sort_values('fiscalDateEnding').reset_index(drop=True)
        existing_ann['fiscalDateEnding']  = existing_ann['fiscalDateEnding'].apply(lambda ts: str(ts))
        existing_quar['fiscalDateEnding'] = existing_quar['fiscalDateEnding'].apply(lambda ts: str(ts))
        
        if cn_updated_quar>0:
            logger.info(f"  Updated quarterly financial statements on {(cn_updated_quar)} rows for ticker {self.asset.ticker}.")
        if cn_updated_ann>0:
            logger.info(f"  Updated annual financial statements on {(cn_updated_ann)} rows for ticker {self.asset.ticker}.")
            
        self.asset.financials_annually = existing_ann
        self.asset.financials_quarterly = existing_quar