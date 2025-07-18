import pandas as pd
from dataclasses import dataclass
from typing import Dict

@dataclass
class AssetData:
    ###########
    # GENERAL #
    ###########
    ticker: str
    isin: str = ""
    
    #Information about the asset in dict format
    about: Dict = None
    # Sector of the asset
    sector: str = ""
    # 'other', 'industrials', 'healthcare', 'technology', 'financial-services', 'real-estate', 'energy', 'consumer-cyclical'

    ###########################
    # PRICES AND CORP-ACTIONS #
    ###########################
    shareprice: pd.DataFrame = None
    # Columns 
    #  'Date'      : str (YYYY-MM-DD)
    #  'Open'      : float
    #  'High'      : float
    #  'Low'       : float
    #  'Close'     : float
    #  'AdjClose'  : float
    #  'Volume'    : float
    #  'Dividends' : float  
    #  'Splits'    : float

    ##############
    # FINANCIALS #
    ##############
    financials_quarterly: pd.DataFrame = None 
    # Columns
    #  'fiscalDateEnding'             : str  (YYYY-MM-DD)
    #  'reportedDate'                 : str  (YYYY-MM-DD)
    #  'reportedEPS'                  : float
    #  'estimatedEPS'                 : float
    #  'surprise'                     : float
    #  'surprisePercentage'           : float
    #  'reportTime'                   : str  ('pre-market', 'post-market')
    #  'grossProfit'                  : float
    #  'totalRevenue'                 : float
    #  'ebit'                         : float
    #  'ebitda'                       : float
    #  'totalAssets'                  : float
    #  'totalCurrentLiabilities'      : float
    #  'totalShareholderEquity'       : float
    #  'commonStockSharesOutstanding' : float
    #  'operatingCashflow'            : float
    
    financials_annually: pd.DataFrame = None
    # Columns
    #  'fiscalDateEnding'             : str  (YYYY-MM-DD)
    #  'reportedEPS'                  : float
    #  'grossProfit'                  : float
    #  'totalRevenue'                 : float
    #  'ebit'                         : float
    #  'ebitda'                       : float
    #  'totalAssets'                  : float
    #  'totalCurrentLiabilities'      : float
    #  'totalShareholderEquity'       : float
    #  'operatingCashflow'            : float

    