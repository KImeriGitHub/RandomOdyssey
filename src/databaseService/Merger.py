import pandas as pd
from datetime import datetime as dt
from typing import Dict, Tuple, List
from src.common.AssetData import AssetData
import logging

logger = logging.getLogger(__name__)

# For Alpha Vantage
class Merger_AV():
    # Columns to exclude from numeric conversion
    _exclude_cols = ['fiscalDateEnding', 'reportedDate', 'reportedCurrency', 'reportTime']

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
            concat = pd.concat([existing, appended], ignore_index=True)
            concat = concat.sort_values('Date').reset_index(drop=True)
            concat['Date'] = concat['Date'].apply(lambda ts: str(ts))
            self.asset.shareprice = concat
            
            logger.info(f"  Added {len(new_rows)} new rows to shareprice data of ticker {self.asset.ticker}.")
        else:
            logger.info(f"  No new rows added to shareprice data of ticker {self.asset.ticker}.")
            
        # print summary
        if diffs:
            logger.info(f"  Changes >1%:\n" + "\n".join(diffs))
        else:
            logger.info(f"  No differences >1% found.")
            
    def merge_financials(self, fin_ann: pd.DataFrame, fin_quar: pd.DataFrame) -> None:
        pass