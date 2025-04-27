import pandas as pd
from datetime import datetime

from src.common.AssetData import AssetData

import logging
logger = logging.getLogger(__name__)

class Checks:
    @staticmethod
    def checkFinanTo(asset: AssetData, year: int) -> bool:
        start: pd.Timestamp = pd.to_datetime(f'{year}-01-01')
        today: pd.Timestamp = pd.to_datetime(datetime.now())
        annual_entries = today.year - start.year
        quarterly_entries = (today.year - start.year) * 4 + (today.month - start.month) // 3 + 1
  
        if asset.financials_quarterly is None:
            return False
        if asset.financials_annually is None:
            return False
        if not asset.financials_quarterly.columns.__contains__('fiscalDateEnding'):
            return False
        if not asset.financials_annually.columns.__contains__('fiscalDateEnding'):
            return False
        if len(asset.financials_annually) < annual_entries:
            return False
        if len(asset.financials_quarterly) < quarterly_entries:
            return False
  
        # Check if the asset has no empty entries in the annual financials
        mask_ann = asset.financials_annually['fiscalDateEnding'].apply(lambda ts: pd.to_datetime(ts)) >= start
        if asset.financials_annually[mask_ann].isnull().any().any():
            return False
  
        # Check if the asset has no empty entries in the quarterly financials except for the last row if date within 30 days
        df = asset.financials_quarterly.copy()
        mask_quar = df['fiscalDateEnding'].apply(lambda ts: pd.to_datetime(ts)) >= start
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        last_date = df['fiscalDateEnding'].max()
        cutoff = today - pd.Timedelta(days=30)
        if last_date >= cutoff:
            # allow NaNs only in the last row
            mask = df['fiscalDateEnding'] == last_date
            df_hist: pd.DataFrame = df.loc[~mask & mask_quar]
            if df_hist.isnull().any().any():
                return False
        else:
            if df[mask_quar].isnull().any().any():
                return False
  
        sp: pd.DataFrame = asset.shareprice.copy()
        sp['Date'] = pd.to_datetime(sp['Date'])
        df = sp[sp['Date'] >= start]
  
        # Check if the shareprice has more than 250*years entries
        if df.empty:
            return False
  
        last: pd.Timestamp = df['Date'].max()
  
        if last < today - pd.Timedelta(days=20):
            logging.warning(f"Last date in shareprice is too old: {last}")
            return False
  
        years = (last - start).days / 365.0
        required = int(250 * years)
  
        if len(df) < required:
            return False
  
        return True
    
    @staticmethod
    def checkOverYear(asset: AssetData, year: int) -> bool:
        start: pd.Timestamp = pd.to_datetime(f'{year}-01-01')
        today: pd.Timestamp = pd.to_datetime(datetime.now())
        sp: pd.DataFrame = asset.shareprice.copy()
        sp['Date'] = pd.to_datetime(sp['Date'])
        df = sp[sp['Date'] >= start]
        
        if df.empty:
            return False
        
        last: pd.Timestamp = df['Date'].max()
        if last < today - pd.Timedelta(days=20):
            logging.warning(f"Last date in shareprice is too old: {last}")
            return False
        
        years = (last - start).days / 365.0
        required = int(250 * years)
        if len(df) < required:
            return False
    
        return True
