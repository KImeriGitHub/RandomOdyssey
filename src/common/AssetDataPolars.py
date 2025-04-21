import polars as pl
from dataclasses import dataclass
from typing import Dict

@dataclass
class AssetDataPolars:
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
    shareprice: pl.DataFrame = None
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
    financials_quarterly: pl.DataFrame = None 
    # Columns
    #  'fiscalDateEnding'             : str  (YYYY-MM-DD)
    #  'reportedDate'                 : str  (YYYY-MM-DD)
    #  'reportedEPS'                  : float
    #  'estimatedEPS'                 : str  (YYYY-MM-DD)
    #  'surprise'                     : float
    #  'surprisePercentage'           : float
    #  'reportTime'                   : str  ('pre_market', 'post-market')
    #  'grossProfit'                  : float
    #  'totalRevenue'                 : float
    #  'ebit'                         : float
    #  'ebitda'                       : float
    #  'totalAssets'                  : float
    #  'totalCurrentLiabilities'      : float
    #  'totalShareholderEquity'       : float
    #  'commonStockSharesOutstanding' : float
    #  'operatingCashflow'            : float
    #  'profitLoss'                   : float
    
    financials_annually: pl.DataFrame = None
    # Columns
    #  'fiscalDateEnding'             : str (YYYY-MM-DD)
    #  'reportedEPS'                  : float
    #  'grossProfit'                  : float
    #  'totalRevenue'                 : float
    #  'ebit'                         : float
    #  'ebitda'                       : float
    #  'totalAssets'                  : float
    #  'totalCurrentLiabilities'      : float
    #  'totalShareholderEquity'       : float
    #  'operatingCashflow'            : float
    #  'profitLoss'                   : float