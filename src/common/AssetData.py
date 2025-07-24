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
    #  'Date'      : str (YYYY-MM-DD)  CANNOT BE NULL
    #  'Open'      : Float64
    #  'High'      : Float64
    #  'Low'       : Float64
    #  'Close'     : Float64
    #  'AdjClose'  : Float64
    #  'Volume'    : Float64
    #  'Dividends' : Float64
    #  'Splits'    : Float64

    ##############
    # FINANCIALS #
    ##############
    financials_quarterly: pd.DataFrame = None 
    # Columns
    #  'fiscalDateEnding'             : str  (YYYY-MM-DD) CANNOT BE NULL
    #  'reportedDate'                 : str  (YYYY-MM-DD) CANNOT BE NULL
    #  'reportedEPS'                  : Float64 
    #  'estimatedEPS'                 : Float64
    #  'surprise'                     : Float64
    #  'surprisePercentage'           : Float64
    #  'reportTime'                   : str  ('pre-market', 'post-market', pd.NA)
    #  'grossProfit'                  : Float64
    #  'totalRevenue'                 : Float64
    #  'ebit'                         : Float64
    #  'ebitda'                       : Float64
    #  'totalAssets'                  : Float64
    #  'totalCurrentLiabilities'      : Float64
    #  'totalShareholderEquity'       : Float64
    #  'commonStockSharesOutstanding' : Float64
    #  'operatingCashflow'            : Float64

    financials_annually: pd.DataFrame = None
    # Columns
    #  'fiscalDateEnding'             : str  (YYYY-MM-DD) CANNOT BE NULL
    #  'reportedEPS'                  : Float64
    #  'grossProfit'                  : Float64
    #  'totalRevenue'                 : Float64
    #  'ebit'                         : Float64
    #  'ebitda'                       : Float64
    #  'totalAssets'                  : Float64
    #  'totalCurrentLiabilities'      : Float64
    #  'totalShareholderEquity'       : Float64
    #  'operatingCashflow'            : Float64

    