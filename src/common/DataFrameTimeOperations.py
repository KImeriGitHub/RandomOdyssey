import pandas as pd
import polars as pl
import numpy as np

class DataFrameTimeOperationsPandas:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        index = self.df.index
        # Check if index is datetime
        is_datetime = pd.api.types.is_datetime64_any_dtype(index)
        if is_datetime:
            is_sorted = index.is_monotonic_increasing
            has_timezone = index.tz is not None

    def getIndex(self, targetDate: pd.Timestamp, timeDelta: pd.Timedelta = pd.Timedelta(days=0.5)) -> int:
        index = self.df.index

        lenIndex = len(index)

        left, right = 0, lenIndex - 1

        while left <= right:
            mid = (left + right) // 2
            currentDate = index[mid]

            # Check if the difference is within the delta
            if abs(currentDate - targetDate) <= timeDelta:
                return mid

            # Adjust the search range
            if currentDate < targetDate:
                left = mid + 1
            else:
                right = mid - 1

        return -1
    
    def getNextLowerOrEqualIndex(self, targetDate: pd.Timestamp) -> int:
        """
            Get the index of the closest date that is lower or equal than the target date
        """
        # Get the index of dates
        dateIndex = self.df.index

        # Check boundary conditions
        if targetDate < dateIndex[0]:
            return -1
        if targetDate >= dateIndex[-1]:
            return len(dateIndex)-1

        # Bisection method to find the closest row
        low, high = 0, len(dateIndex) - 1
        while low <= high:
            mid = (low + high) // 2
            if dateIndex[mid] == targetDate:
                return mid
            elif dateIndex[mid] < targetDate:
                low = mid + 1
            else:
                high = mid - 1

        # The closest smaller date is at index `high` after the loop
        return high

    def inbetween(self,
                  startDate: pd.Timestamp, 
                  endDate: pd.Timestamp,
                  timeDelta: pd.Timedelta = pd.Timedelta(days=0)):
        """
            Returns dataframe of slice with dates between and including the start and end date
        """
        if startDate > endDate:
            return ValueError("Start Date is later than the End Date!")

        startIdx = self.getNextLowerOrEqualIndex(startDate-timeDelta)
        endIdx = self.getNextLowerOrEqualIndex(endDate+timeDelta)

        return self.df.iloc[startIdx:endIdx+1]
    
    def around(self,
            date: pd.Timestamp,
            timeDelta: pd.Timedelta = pd.Timedelta(days=0.5)):

        idx = self.getIndex(date, timeDelta)

        if idx == -1:
            return pd.DataFrame(None)

        return self.df.iloc[idx]
    
class DataFrameTimeOperationsPolars:
    def __init__(self, df: pl.DataFrame, dateCol: str = 'Date'):
        self.df = df
        self.index = self.df[dateCol]
        if self.index.dtype.time_zone is None:
            self.index = self.index.dt.replace_time_zone("UTC")
        # Check if index is datetime
        is_sorted = self.index.is_sorted()
        
        if not is_sorted:
            return ValueError("Dates are not sorted!")

    def getIndex(self, targetDate: pd.Timestamp, timeDelta: pd.Timedelta = pd.Timedelta(days=0.5)) -> int:
        lenIndex = len(self.index)

        left, right = 0, lenIndex - 1

        while left <= right:
            mid = (left + right) // 2
            currentDate = self.index[mid]

            # Check if the difference is within the delta
            if abs(currentDate - targetDate) <= timeDelta:
                return mid

            # Adjust the search range
            if currentDate < targetDate:
                left = mid + 1
            else:
                right = mid - 1

        return -1
    
    def getNextLowerOrEqualIndex(self, targetDate: pd.Timestamp) -> int:
        # Get the index of dates
        dateIndex = self.index
        
        # Check boundary conditions
        if targetDate < dateIndex[0]:
            return -1
        if targetDate >= dateIndex[-1]:
            return len(dateIndex)-1

        # Bisection method to find the closest row
        low, high = 0, len(dateIndex) - 1
        while low <= high:
            mid = (low + high) // 2
            if dateIndex[mid] == targetDate:
                return mid
            elif dateIndex[mid] < targetDate:
                low = mid + 1
            else:
                high = mid - 1

        # The closest smaller date is at index `high` after the loop
        return high

    def inbetween(self,
                  startDate: pd.Timestamp, 
                  endDate: pd.Timestamp,
                  timeDelta: pd.Timedelta = pd.Timedelta(days=0)):
        """
            Returns dataframe of slice with dates between and including the start and end date
        """
        if startDate > endDate:
            return ValueError("Start Date is later than the End Date!")

        startIdx = self.getNextLowerOrEqualIndex(startDate-timeDelta)
        endIdx = self.getNextLowerOrEqualIndex(endDate+timeDelta)

        return self.df.slice(startIdx,endIdx-startIdx+1)
    
    def around(self,
            date: pd.Timestamp,
            timeDelta: pd.Timedelta = pd.Timedelta(days=0.5)):

        idx = self.getIndex(date, timeDelta)

        if idx == -1:
            return pl.DataFrame(None)

        return self.df.row[idx]

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
