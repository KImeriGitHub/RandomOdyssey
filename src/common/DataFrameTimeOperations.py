import pandas as pd
import polars as pl

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
    
    def getNextLowerIndex(self, targetDate: pd.Timestamp) -> int:
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

        if startDate > endDate:
            return ValueError("Start Date is later than the End Date!")

        startIdx = self.getNextLowerIndex(startDate-timeDelta) + 1
        endIdx = self.getNextLowerIndex(endDate+timeDelta)+1

        return self.df.iloc[startIdx:endIdx]
    
    def around(self,
            date: pd.Timestamp,
            timeDelta: pd.Timedelta = pd.Timedelta(days=0.5)):

        idx = self.getIndex(date, timeDelta)

        if idx == -1:
            return pd.DataFrame(None)

        return self.df.iloc[idx]
    
class DataFrameTimeOperationsPolars:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.index = self.df['Date']
        if self.index.dtype.time_zone is None:
            self.index = self.index.dt.replace_time_zone("UTC")
        # Check if index is datetime
        is_sorted = self.index.is_sorted()

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
    
    def getNextLowerIndex(self, targetDate: pd.Timestamp) -> int:
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

        if startDate > endDate:
            return ValueError("Start Date is later than the End Date!")

        startIdx = self.getNextLowerIndex(startDate-timeDelta) + 1
        endIdx = self.getNextLowerIndex(endDate+timeDelta)+1

        return self.df.slice(startIdx,endIdx-startIdx)
    
    def around(self,
            date: pd.Timestamp,
            timeDelta: pd.Timedelta = pd.Timedelta(days=0.5)):

        idx = self.getIndex(date, timeDelta)

        if idx == -1:
            return pl.DataFrame(None)

        return self.df.row[idx]