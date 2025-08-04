import numpy as np
import pandas as pd
from typing import List
import datetime
import holidays

from src.common.AssetDataPolars import AssetDataPolars

class FeatureSeasonal():
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'monthsHorizon': 12,
        'timesteps': 10,
        'lagList': [1, 2, 5, 10, 20, 50, 100, 200, 300, 500],
    }
    
    def __init__(self, 
            asset: AssetDataPolars, 
            startDate: datetime.date, 
            endDate: datetime.date, 
            params: dict = None
        ):
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.monthsHorizon = self.params['monthsHorizon']
        self.timesteps = self.params['timesteps']
        self.lagList = self.lagList['lagList']

        self.startDate = startDate-pd.Timedelta(days=max(self.lagList, default=0))
        self.endDate = endDate
        self.asset = asset
        
        self.holidate_dates: list[datetime.date] = self.__USHolidays()
        
    def __USHolidays(self) -> list[pd.Timestamp]:
        country_holidays = holidays.CountryHoliday('US')
        for y in range(self.startDate.year-1, self.endDate.year+2):
            country_holidays.get(f"{y}")
        country_holidays = sorted(country_holidays.keys())
        return [pd.Timestamp(val.__str__(), tz= 'UTC').date() for val in country_holidays]
        
    
    def getFeatureNames(self) -> list[str]:
        features_names = [
            "Seasonal_year",
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
        
        for lag in self.lagList:
            features_names.extend([
                #f"Seasonal_year_lag_m{lag}",
                #f"Seasonal_month_lag_m{lag}",
                #f"Seasonal_day_lag_m{lag}",
                #f"Seasonal_day_of_week_lag_m{lag}",
                #f"Seasonal_quarter_lag_m{lag}",
                f"Seasonal_week_of_year_lag_m{lag}",
                #f"Seasonal_is_month_start_lag_m{lag}",
                #f"Seasonal_is_month_end_lag_m{lag}",
                #f"Seasonal_is_year_start_lag_m{lag}",
                #f"Seasonal_is_year_end_lag_m{lag}",
                #f"Seasonal_week_part_lag_m{lag}",
                f"Seasonal_days_to_next_holiday_lag_m{lag}",
                f"Seasonal_days_since_last_holiday_lag_m{lag}",
            ])
            
        return features_names
    
    def getTimeFeatureNames(self) -> list[str]:
        res_names = []
        
        res_names += ["Seasonal_day_of_week"]
        res_names += ["Seasonal_week_of_year"]
        res_names += ["Seasonal_days_to_nearest_holiday"]
        
        return res_names
    
    def apply(self, date: datetime.date, scaleToNiveau: float):
        """
        Extracts comprehensive date-related features for a given date.
        """
        
        # General date-related features
        features_raw = np.array([
            date.year-self.startDate.year,
            date.month-1,
            date.day-1,
            date.weekday(),  # Monday=0, Sunday=6
            (date.month - 1) // 3,
            date.isocalendar()[1]-1,  # Week number of the year
            date.day < 7,
            date.day > 21,
            date.month < 3,
            date.month > 10,
            (0 if date.weekday() < 2 else 1 if date.weekday() < 4 else 2),
            np.max([np.min([((h) - (date)).days for h in self.holidate_dates if (h) >= (date)]),90]),
            np.max([np.min([((date) - (h)).days for h in self.holidate_dates if (h) <= (date)]),90]),
        ])
        
        unifyingFactorArray = np.array([
            1/(self.endDate.year - self.startDate.year+1),
            1/11.0, 
            1/(pd.Timestamp(date).days_in_month-1), 
            1/6.0, 
            1/3.0, 
            1/51.0, 
            1.0, 
            1.0, 
            1.0, 
            1.0, 
            1/2.0, 
            1/90.0, 
            1/90.0
        ])
        features_raw = features_raw * unifyingFactorArray
        
        for lag in self.lagList:
            date_lag = date - pd.Timedelta(days=lag)
            features_raw_lag = np.array([
                date_lag.year-self.startDate.year,
                (date_lag.month-1)*(pd.Timestamp(date_lag).days_in_month-1)/(pd.Timestamp(date_lag).days_in_month-1),
                date_lag.day-1,
                date_lag.weekday(),  # Monday=0, Sunday=6
                (date_lag.month - 1) // 3,
                date_lag.isocalendar()[1]-1,  # Week number of the year
                date_lag.day < 7,
                date_lag.day > 10,
                date_lag.month < 3,
                date_lag.month > 10,
                (0 if date_lag.weekday() < 2 else 1 if date_lag.weekday() < 4 else 2),
                np.max([np.min([((h) - (date_lag)).days for h in self.holidate_dates if (h) >= (date_lag)]), 90]),
                np.max([np.min([((date_lag) - (h)).days for h in self.holidate_dates if (h) <= (date_lag)]), 90]),
            ])
            features_raw_lag = features_raw_lag * unifyingFactorArray
            # Selected features 
            features_raw_lag = features_raw_lag[[5,11,12]]
            
            features_raw = np.concatenate([features_raw, features_raw_lag])
            
        
        features = features_raw * scaleToNiveau
        return features.astype(np.float32)
    
    def apply_timeseries(self, date: datetime.date) -> np.ndarray:
        coreLen = len(self.getTimeFeatureNames())
        featuresMat = np.zeros((self.timesteps, coreLen))
        
        for ts in range(0, self.timesteps):
            date_lag = date - pd.Timedelta(days = ((self.timesteps - 1) - ts))
            
            featuresMat[ts, 0] = date_lag.weekday()/6.0 # Monday=0, Sunday=6
            featuresMat[ts, 1] = np.clip((date_lag.isocalendar()[1]-1)/52.0, 0,1) # Week number of the year
            featuresMat[ts, 2] = np.tanh(min([((date_lag)-(h)).days for h in self.holidate_dates], key=lambda x: (abs(x), -x)))/2.0+0.5
            
        return featuresMat.astype(np.float32)
        
        
        
        
    
""" 
For future non-US uses, the following might help:

pycountry.countries.lookup(asset.about.get('country','United States')).alpha_2 

This was for yfinance. Needs to be checked for all possible tickers. doesnt work for turkey
"""