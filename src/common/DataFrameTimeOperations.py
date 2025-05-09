import pandas as pd
import polars as pl
import numpy as np
import datetime
import bisect
from typing import Iterable, Optional
    
class DataFrameTimeOperations:
    def __init__(self, df: pl.DataFrame, dateCol: str = 'Date'):
        self.df = df
        if dateCol not in self.df.columns:
            raise ValueError(f"Column '{dateCol}' does not exist in the DataFrame.")
        
        # parse/cast to Polars Date
        dtype = self.df[dateCol].dtype
        if dtype == pl.Utf8:
            try:
                self.df = self.df.with_columns(
                    pl.col(dateCol).str.strptime(pl.Date, format=None).alias(dateCol)
                )
            except Exception as e:
                raise ValueError(f"Could not parse strings in '{dateCol}' as dates: {e}")
        else:
            if dtype != pl.Date:
                raise ValueError(f"Column '{dateCol}' must be of type Date or Utf8, got {dtype}.")
        
        self.series = self.df[dateCol]
        if not self.series.is_sorted():
            raise ValueError("Dates are not sorted!")
        self.len = self.series.len()

    def getIndex(self, targetDate: datetime.date) -> int | None:
        """
        Return the exact‐match index of targetDate, or None if not present.
        """
        i = self.series.search_sorted(targetDate, 'left')
        if i == self.len:
            return None
        if not self.series[i] == targetDate:
            return None
        return i

    def getNextLowerOrEqualIndex(self, targetDate: datetime.date) -> int:
        """
        Return the largest index i such that self.index[i] <= targetDate.
        """
        i = self.series.search_sorted(targetDate, 'left')
        if i == self.len:
            i = i-1
            
        if self.series[i] == targetDate:
            return i
        if self.series[i] < targetDate:
            return i
        
        return i-1

    def inbetween(self,
                  startDate: datetime.date, 
                  endDate: datetime.date) -> list[int]:
        """
        Returns list of integer positions for dates between (inclusive) startDate and endDate.
        """
        if startDate > endDate:
            raise ValueError("start must be ≤ end")
        lo = self.series.search_sorted(startDate, 'left')-1
        hi = self.series.search_sorted(endDate, 'right')
        # include end if exact match
        if hi < self.len and self.series[hi] == endDate:
            hi += 1
        return list(range(lo, hi))
    
    def getIndices(self, targetDates: Iterable[datetime.date]) -> list[Optional[int]]:
        """
        For each date in targetDates, return its exact index or None if not present.
        """
        idcs = self.series.search_sorted(targetDates, 'left')
        
        # fix idx out of range of series
        idcs = [idx-1 if idx == self.len else idx for idx in idcs]
        
        # check whether it is an exact match
        idcs = [idx if self.series[idx] == targetDates[i] else None for i, idx in enumerate(idcs)]
        
        return idcs

    def getNextLowerOrEqualIndices(self, targetDates: Iterable[datetime.date]) -> list[int]:
        """
        For each date in targetDates, return the largest index i such that index[i] ≤ date.
        """
        idcs = self.series.search_sorted(targetDates, 'left')
        
        # fix idx out of range of series
        idcs = [idx-1 if idx == self.len else idx for idx in idcs]
        
        # adjust for search_sorted behavior
        idcs = [idx-1 if self.series[idx] > targetDates[i] else idx for i, idx in enumerate(idcs)]
        
        return idcs

class FastTimeOpsPolars:
    def __init__(self, df: pl.DataFrame, date_col: str = 'Date'):
        self.df = df
        # extract datetime values into pandas Series for tz handling
        ts_list = df[date_col].to_list()
        s = pd.to_datetime(pd.Series(ts_list))
        # ensure timezone-aware, default UTC
        if s.dt.tz is None:
            s = s.dt.tz_localize('UTC')
        # store int64 nanoseconds array
        self._ts = s.view('int64').to_numpy()
        # ensure sorted
        if not np.all(self._ts[:-1] <= self._ts[1:]):
            raise ValueError("Date column must be sorted")

    def inbetween(self, start: pd.Timestamp, end: pd.Timestamp) -> pl.DataFrame:
        # ensure pd.Timestamp with tz
        if start.tzinfo is None:
            start = start.tz_localize('UTC')
        if end.tzinfo is None:
            end = end.tz_localize('UTC')
        # convert to ns
        s_ns = start.value
        e_ns = end.value
        # find bounds
        i = np.searchsorted(self._ts, s_ns, side='left')
        j = np.searchsorted(self._ts, e_ns, side='right') - 1
        if i > j:
            return self.df.slice(0, 0)
        return self.df.slice(i, j - i + 1)

    def around(self, target: pd.Timestamp, tol: pd.Timedelta = pd.Timedelta(days=0.5)) -> pl.DataFrame:
        # ensure pd.Timestamp with tz
        if target.tzinfo is None:
            target = target.tz_localize('UTC')
        # convert to ns
        t0_ns = target.value
        tol_ns = tol.value
        # compute window
        lo = np.searchsorted(self._ts, t0_ns - tol_ns, side='left')
        hi = np.searchsorted(self._ts, t0_ns + tol_ns, side='right')
        if lo >= hi:
            return self.df.slice(0, 0)
        return self.df.slice(lo, hi - lo)
