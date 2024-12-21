import numpy as np
import pandas as pd
import polars as pl
from typing import Dict
import holidays

from src.common.AssetDataPolars import AssetDataPolars

class FeatureSeasonal():
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'monthsHorizon': 12,
    }
    
    def __init__(self, asset: AssetDataPolars, startDate: pd.Timestamp, endDate:pd.Timestamp, params: dict = None):
        self.startDate = startDate
        self.endDate = endDate
        self.asset = asset
        
        # Update default parameters with any provided parameters
        self.params = self.DEFAULT_PARAMS
        if params is not None:
            self.params.update(params)

        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.monthsHorizon = self.params['monthsHorizon']
        
        self.holidate_dates = self.__USHolidays()
        
    def __USHolidays(self):
        country_holidays = holidays.CountryHoliday('US')
        for y in range(self.startDate.year-1, self.endDate.year+2):
            country_holidays.get(f"{y}")
        country_holidays = sorted(country_holidays.keys())
        return [pd.Timestamp(val.__str__(), tz= 'UTC') for val in country_holidays]
        
    
    def getFeatureNames(self) -> list[str]:
        features_names = [
            "Seasonal_month",
            "Seasonal_day",
            "Seasonal_day_of_week",  # Monday=0, Sunday=6
            "Seasonal_quarter",
            "Seasonal_week_of_year",  # Week number of the year
            "Seasonal_is_month_start",
            "Seasonal_is_month_end",
            "Seasonal_is_year_start",
            "Seasonal_is_year_end",
            "Seasonal_week_part",
            "Seasonal_days_to_next_holiday",
            "Seasonal_days_since_last_holiday",
        ]
            
        return features_names
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float):
        """
        Extracts comprehensive date-related features for a given pd.Timestamp.
        Parameters:
            timestamp (pd.Timestamp): The date to extract features from.
        """
        
        if not isinstance(date, pd.Timestamp):
            raise ValueError("The input must be a pandas Timestamp object.")
        # Ensure timestamp is timezone-aware (if not already)
        date = date.tz_localize('UTC') if date.tz is None else date
        
        # General date-related features
        features_raw = np.array([
            date.month-1,
            date.day-1,
            date.dayofweek,  # Monday=0, Sunday=6
            date.quarter-1,
            date.isocalendar()[1]-1,  # Week number of the year
            date.is_month_start,
            date.is_month_end,
            date.is_year_start,
            date.is_year_end,
            (0 if date.dayofweek < 2 else 1 if date.dayofweek < 4 else 2),
            np.max([np.min([(h - date).days for h in self.holidate_dates if h >= date]),90]),
            np.max([np.min([(date - h).days for h in self.holidate_dates if h <= date]),90]),
        ])
        
        unifyingFactorArray = np.array([1/11.0, 1/(date.days_in_month-1), 1/6.0, 1/3.0, 1/51.0, 1.0, 1.0, 1.0, 1.0, 1/2.0, 1/90.0, 1/90.0])
        
        features = features_raw * unifyingFactorArray * scaleToNiveau
        
        return features
    
    
""" 
For future non-US uses, the following might help:

pycountry.countries.lookup(asset.about.get('country','United States')).alpha_2 

This was for yfinance. Needs to be checked for all possible tickers. doesnt work for turkey
"""