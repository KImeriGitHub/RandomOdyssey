import numpy as np
import pandas as pd
from typing import List
import datetime
import holidays

from src.featureAlchemy.IFeature import IFeature
from src.common.AssetDataPolars import AssetDataPolars

class FeatureSeasonalTS(IFeature):
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
        self.lagList = self.params['lagList']

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
        res_names = []
        
        res_names += ["Seasonal_day_of_week"]
        res_names += ["Seasonal_week_of_year"]
        res_names += ["Seasonal_days_to_nearest_holiday"]
        
        return res_names

    def apply(self, dates: List[datetime.date]) -> np.ndarray:
        # params
        ts = self.timesteps
        names = self.getFeatureNames()
        n_feat = len(names)
        
        # 1) convert input dates to numpy datetime64[D]
        dates_arr = np.array(dates, dtype='datetime64[D]')                    # (n,)
        n = dates_arr.shape[0]

        # 2) build an array of lags: [timesteps‑1 days ago, ..., today]
        lags = np.arange(ts-1, -1, -1, dtype='timedelta64[D]')               # (ts,)
        
        # 3) broadcast to get all date_lags: shape (n, ts)
        date_lags = dates_arr[:, None] - lags[None, :]                        # (n,ts)

        # 4) feature 1: normalized weekday
        wds = np.array(pd.DatetimeIndex(date_lags.flatten()).weekday)                  # 0=Mon…6=Sun
        wd_norm = (wds / 6.0).reshape(n, ts)                                 # (n,ts)

        # 5) feature 2: normalized ISO week number
        weeks = pd.DatetimeIndex(date_lags.flatten()).isocalendar().week      # 1–53
        wk_norm = np.clip((weeks-1)/52.0, 0, 1).to_numpy().reshape(n, ts)     # (n,ts)

        # 6) feature 3: proximity to nearest holiday
        hols = np.array(self.holidate_dates, dtype='datetime64[D]')          # (h,)
        diffs = date_lags[:, :, None] - hols[None, None, :]                  # (n,ts,h)
        dd = diffs.astype('timedelta64[D]').astype(int)                      # (n,ts,h)
        abs_dd = np.abs(dd)
        m = abs_dd.min(axis=2)                                               # (n,ts)
        # tie‑break: pick the positive diff if equidistant
        masked = np.where(abs_dd == m[:,:,None], dd, -np.inf)                # (n,ts,h)
        nearest = masked.max(axis=2)                                         # (n,ts)
        hol_norm = (np.tanh(nearest) / 2.0 + 0.5)                            # (n,ts)

        # 7) stack into final feature tensor
        feat = np.stack([wd_norm, wk_norm, hol_norm], axis=2)                # (n,ts,3)

        return feat.astype(np.float32)

""" 
For future non-US uses, the following might help:

pycountry.countries.lookup(asset.about.get('country','United States')).alpha_2 

This was for yfinance. Needs to be checked for all possible tickers. doesnt work for turkey
"""