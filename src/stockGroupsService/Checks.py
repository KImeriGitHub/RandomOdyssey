import pandas as pd
import numpy as np
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
        df['reportedDate'] = pd.to_datetime(df['reportedDate'])
        last_date = df['reportedDate'].max()
        cutoff = today - pd.Timedelta(days=70) # 2 months + buffer for delayed informations
        if last_date >= cutoff:
            # allow NaNs only in the last row
            mask_last_day = df['reportedDate'] == last_date
            df_hist: pd.DataFrame = df.loc[~mask_last_day & mask_quar]
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
    
    @staticmethod
    def is_regular_ohlcv(asset:AssetData) -> bool:
        sp: pd.DataFrame = asset.shareprice.copy()
        if sp.empty:
            return False
        
        sp['Date'] = pd.to_datetime(sp['Date'])
        start: pd.Timestamp = sp['Date'].min()
        end: pd.Timestamp = sp['Date'].max()

        df = sp[sp['Date'] >= start]
        if df.empty:
            return False
        
        # Check whether there are 250 entries every year
        years_adj = (end - start).days / 365.0
        required = int(250 * years_adj)
        if len(df) < required:
            return False

        # Check that OHLC does not repeat over 3 times
        cols = ["Open","High","Low","Close"]
        v = df[cols].to_numpy(dtype=np.float64)
        if np.isnan(v[1:-1]).any():
            return False
        def is_close(a, b, rtol=1e-10, atol=0, equal_nan=False):
            return np.all(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan), axis=1)
        any_consec_dub = (
            np.any(
                is_close(v[:-3], v[3:]) 
                & is_close(v[1:-2], v[2:-1]) 
                & is_close(v[2:-1], v[1:-2])
            )
        )
        if any_consec_dub:
            return False

        # Check whether adjusted close return over a day is between 1e-5 and 1e5
        adj_close = df["AdjClose"]
        ret_ratio = adj_close / adj_close.shift(1)
        # skip the first NaN and check the range
        upper_bound = 1e2
        lower_bound = 1e-2
        if not ret_ratio.iloc[1:].between(lower_bound, upper_bound).all():
            return False

        # Volume non-zero except possibly the last two rows
        last_idx = df.index[-1]
        if ((df["Volume"] == 0) & (df.index != last_idx) & (df.index != last_idx - 1)).any():
            return False

        return True