from abc import ABC, abstractmethod
from typing import Dict
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
import pandas as pd

class IStrategy(ABC):
    @abstractmethod
    def apply(self,
              assets: Dict[str, AssetData], 
              portfolio: Portfolio, 
              currentDate: pd.Timestamp, 
              assetdateIdx: Dict[str, int] = {}):
        """Apply the strategy to the given assets and update the portfolio accordingly.
        
        assetdateIdx: Index of assets at the closest trading day to currentDate."""
        pass
