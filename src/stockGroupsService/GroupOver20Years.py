import pandas as pd
from datetime import datetime

from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup

import logging
logger = logging.getLogger(__name__)

class GroupOver20Years(IGroup):
    def groupName(self) -> str:
        return "group_over20years"

    def checkAsset(self, asset: AssetData) -> bool:
        start: pd.Timestamp = pd.to_datetime('2004-01-01')
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
