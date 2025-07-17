import pandas as pd
import polars as pl
import numpy as np
import datetime
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
        lo = self.series.search_sorted(startDate, 'left')
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
        HAS A -1 IF FIRST INDEX IS GREATER THAN TARGET DATE
        """
        idcs = self.series.search_sorted(targetDates, 'left')
        
        # fix idx out of range of series
        idcs = [idx-1 if idx == self.len else idx for idx in idcs]
        
        # adjust for search_sorted behavior
        idcs = [idx-1 if self.series[idx] > targetDates[i] else idx for i, idx in enumerate(idcs)]
        
        return idcs