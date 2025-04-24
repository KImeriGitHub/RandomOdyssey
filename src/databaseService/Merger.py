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
        fullSharePrice['Date'] = fullSharePrice['Date'].apply(lambda ts: ts.date())
        
        full = fullSharePrice.copy()
        existing = self.asset.shareprice.copy()
        existing['Date'] = existing['Date'].apply(lambda ts: dt.strptime(ts, '%Y-%m-%d').date())
        
        new_rows = []
        diffs = []
        cols = ['Open','High','Low','Close','AdjClose','Volume','Dividends','Splits']

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
        assert pd.api.types.is_datetime64_any_dtype(fin_ann['fiscalDateEnding']), (
            "fin_ann.fiscalDateEnding must be datetime64[ns], "
            f"got {fin_ann['fiscalDateEnding'].dtype}"
        )
        assert pd.api.types.is_datetime64_any_dtype(fin_quar['fiscalDateEnding']), (
            "fin_quar.fiscalDateEnding must be datetime64[ns], "
            f"got {fin_quar['fiscalDateEnding'].dtype}"
        )
        full_ann = fin_ann.copy()
        full_ann['fiscalDateEnding'] = full_ann['fiscalDateEnding'].apply(lambda ts: ts.date())
        full_quar = fin_quar.copy()
        full_quar['fiscalDateEnding'] = full_quar['fiscalDateEnding'].apply(lambda ts: ts.date())
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

        is_updated_ann = False
        for _, new in full_ann.iterrows():
            date: dt.date = new['fiscalDateEnding']
            mask = existing_ann['fiscalDateEnding'] == date
            if mask.any():
                ex_idx = existing_ann[mask].index[0]
                for col in existing_ann.columns.drop('fiscalDateEnding'):
                    if pd.isna(existing_ann.at[ex_idx, col]):
                        existing_ann.at[ex_idx, col] = new[col]
                        is_updated_ann = True
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
                        existing_ann = pd.concat([existing_ann, pd.DataFrame([new], columns=existing_ann.columns)], ignore_index=True)
                    is_updated_ann = True
                else:
                    # 2b) some entry in that year → log and skip
                    logger.info(f"  DB annual fiscal date differs from new date. Year {date.year}.")
        
        today = dt.now().date()
        is_updated_quar = False
        for _, new in full_quar.iterrows():
            date = new['fiscalDateEnding']
            mask = existing_quar['fiscalDateEnding'] == date
            if mask.any():
                for col in existing_quar.columns.drop('fiscalDateEnding'):
                    if pd.isna(existing_quar.loc[mask, col].iloc[0]):
                        existing_quar.loc[mask, col] = new[col]
                        is_updated_quar = True
            else:
                age = (today - date).days
                if 0 <= age <= 31:
                    if existing_quar.empty:
                        existing_quar = pd.DataFrame([new], columns=existing_quar.columns)
                    elif pd.DataFrame([new]).empty:
                        pass
                    else:
                        existing_quar = pd.concat([existing_quar, pd.DataFrame([new], columns=existing_quar.columns)], ignore_index=True)
                    logger.info(f"  New quarterly fiscal statement in last month.")
                    is_updated_quar = True
                    
                else:
                    q = (date.month - 1) // 3 + 1
                    mask_in_q = (existing_quar['fiscalDateEnding'].apply(lambda x: x.year) == date.year) & \
                           (existing_quar['fiscalDateEnding'].apply(lambda x: (x.month - 1) // 3 + 1) == q)
                    if not mask_in_q.any():
                        if existing_quar.empty:
                            existing_quar = pd.DataFrame([new], columns=existing_quar.columns)
                        elif pd.DataFrame([new]).empty:
                            pass
                        else:
                            existing_quar = pd.concat([existing_quar, pd.DataFrame([new], columns=existing_quar.columns)], ignore_index=True)
                        is_updated_quar = True
                    else:
                        logger.info(f"  DB quarterly fiscal date differs from new date. Year {date.year} month {date.month}.")
            
        existing_ann = existing_ann.sort_values('fiscalDateEnding').reset_index(drop=True)
        existing_quar = existing_quar.sort_values('fiscalDateEnding').reset_index(drop=True)
        existing_ann['fiscalDateEnding']  = existing_ann['fiscalDateEnding'].apply(lambda ts: str(ts))
        existing_quar['fiscalDateEnding'] = existing_quar['fiscalDateEnding'].apply(lambda ts: str(ts))
        
        if is_updated_quar:
            logger.info(f"  Updated {len(existing_quar)} quarterly financial statements for ticker {self.asset.ticker}.")
        if is_updated_ann:
            logger.info(f"  Updated {len(existing_ann)} annual financial statements for ticker {self.asset.ticker}.")
            
        self.asset.financials_annually = existing_ann
        self.asset.financials_quarterly = existing_quar