from typing import Dict
import pandas as pd
import re
import datetime
from typing import Dict
from pandas.api.types import is_numeric_dtype, is_string_dtype

from src.databaseService.CleanData import CleanData

import logging
logger = logging.getLogger(__name__)

# For Alpha Vantage
class Parser_AV():
    """
        Parser for Alpha Vantage data.
        Must return formats given by AssetData.
    """
    def __init__(self, 
            sharepriceData: pd.DataFrame = pd.DataFrame(None),
            overview = None,
            incStatementData: Dict = {}, 
            cashFlowData: Dict = {}, 
            balanceSheetData: Dict = {}, 
            earningsData: Dict = {},
        ):
        self.sharepriceData = sharepriceData
        self.overview = overview
        self.incStatementData = incStatementData
        self.cashFlowData = cashFlowData
        self.balanceSheetData = balanceSheetData
        self.earningsData = earningsData
        
    def parse_overview(self) -> tuple[Dict, str]:
        if self.overview is None:
            logger.error("Parser: Overview data is empty.")
            raise ValueError("Overview data is empty.")
        
        about = self.overview.to_dict(orient='records')[0]
        catDict = {
            'OTHER': 'other', 
            'MANUFACTURING':'industrials', 
            'LIFE SCIENCES': 'healthcare', 
            'TECHNOLOGY': 'technology', 
            'FINANCE': 'financial-services', 
            'REAL ESTATE & CONSTRUCTION':'real-estate', 
            'ENERGY & TRANSPORTATION': 'energy', 
            'TRADE & SERVICES': 'consumer-cyclical', 
        }
        sector = catDict[self.overview["Sector"].iloc[0]] 
        
        return about, sector
        
    def parse_shareprice(self) -> pd.DataFrame:
        if self.sharepriceData.empty:
            logger.error("Parser: Shareprice data is empty.")
            raise ValueError("Shareprice data is empty.")
        
        ### PREPARE MERGING SHAREPRICE DATA
        fullSharePrice = self.sharepriceData.copy()
        fullSharePrice.index.name = "date"
        fullSharePrice = fullSharePrice.iloc[::-1] #flip upside down
        fullSharePrice.reset_index(inplace=True)
        fullSharePrice.rename(columns={
            'date': 'Date',
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'AdjClose',
            '6. volume': 'Volume',
            '7. dividend amount': 'Dividends',
            '8. split coefficient': 'Splits'
        }, inplace=True)
        cols = ['Open','High','Low','Close','AdjClose','Volume','Dividends','Splits']
        
        # Drop missing dates
        if pd.isnull(fullSharePrice['Date']).any() or fullSharePrice['Date'].isna().any():
            logger.info(f"  Shareprice has empty dates or NaT Dates.")
            fullSharePrice = fullSharePrice[pd.notnull(fullSharePrice['Date'])] # remove empty Date columns
            fullSharePrice = fullSharePrice[fullSharePrice['Date'].notna()] # remove NaT Date columns
            
        # Convert datetime to date
        fullSharePrice['Date'] = fullSharePrice['Date'].apply(lambda ts: ts.date())
        
        # Drop missing values, except for last row
        if fullSharePrice[cols].iloc[:-1].isnull().values.any():
            logger.info("  Shareprice has empty Number values or NaN Number values.")
            # build a mask: keep rows where *all* cols are notna, or it'’'s the last row
            last_idx = fullSharePrice.index[-1]
            mask = fullSharePrice[cols].notna().all(axis=1) | (fullSharePrice.index == last_idx)
            fullSharePrice = fullSharePrice[mask]
            
        # Convert date to str (as of definition AssetData)
        fullSharePrice['Date'] = fullSharePrice['Date'].apply(lambda ts: ts.strftime("%Y-%m-%d"))

        # Asses dtypes
        fullSharePrice = fullSharePrice.astype({
            'Date': 'string',
            'Open': 'Float64', 
            'High': 'Float64',
            'Low': 'Float64', 
            'Close': 'Float64',
            'AdjClose': 'Float64',
            'Volume': 'Float64',
            'Dividends': 'Float64',
            'Splits': 'Float64'
        })

        # Validate data
        self.validate_shareprice(fullSharePrice)
        
        return fullSharePrice
    
    def parse_financials(self):
        if (
            self.incStatementData == {} or
            self.cashFlowData == {} or
            self.balanceSheetData == {} or
            self.earningsData == {}
        ):
            raise ValueError("Parser: Financial Dataframes are empty.")
        
        # Cast to Dataframe
        df_ann_incStat = pd.DataFrame(self.incStatementData['annualReports'])
        df_quar_incStat = pd.DataFrame(self.incStatementData['quarterlyReports'])
        
        df_ann_cashFlow = pd.DataFrame(self.cashFlowData['annualReports'])
        df_quar_cashFlow = pd.DataFrame(self.cashFlowData['quarterlyReports'])
        
        df_ann_balSheet = pd.DataFrame(self.balanceSheetData['annualReports'])
        df_quar_balSheet = pd.DataFrame(self.balanceSheetData['quarterlyReports'])
        
        df_ann_earnings = pd.DataFrame(self.earningsData['annualEarnings'])
        df_quar_earnings = pd.DataFrame(self.earningsData['quarterlyEarnings'])

        # Cast string 'None' to pd.NA
        df_ann_incStat.replace('None', pd.NA, inplace=True)
        df_quar_incStat.replace('None', pd.NA, inplace=True)
        df_ann_cashFlow.replace('None', pd.NA, inplace=True)
        df_quar_cashFlow.replace('None', pd.NA, inplace=True)
        df_ann_balSheet.replace('None', pd.NA, inplace=True)
        df_quar_balSheet.replace('None', pd.NA, inplace=True)
        df_ann_earnings.replace('None', pd.NA, inplace=True)
        df_quar_earnings.replace('None', pd.NA, inplace=True)

        # Cast fiscalDateEnding to datetime
        df_ann_incStat['fiscalDateEnding']   = pd.to_datetime(df_ann_incStat['fiscalDateEnding'], errors='coerce').dt.date
        df_quar_incStat['fiscalDateEnding']  = pd.to_datetime(df_quar_incStat['fiscalDateEnding'], errors='coerce').dt.date
        df_ann_cashFlow['fiscalDateEnding']  = pd.to_datetime(df_ann_cashFlow['fiscalDateEnding'], errors='coerce').dt.date
        df_quar_cashFlow['fiscalDateEnding'] = pd.to_datetime(df_quar_cashFlow['fiscalDateEnding'], errors='coerce').dt.date
        df_ann_balSheet['fiscalDateEnding']  = pd.to_datetime(df_ann_balSheet['fiscalDateEnding'], errors='coerce').dt.date
        df_quar_balSheet['fiscalDateEnding'] = pd.to_datetime(df_quar_balSheet['fiscalDateEnding'], errors='coerce').dt.date
        df_ann_earnings['fiscalDateEnding']  = pd.to_datetime(df_ann_earnings['fiscalDateEnding'], errors='coerce').dt.date
        df_quar_earnings['fiscalDateEnding'] = pd.to_datetime(df_quar_earnings['fiscalDateEnding'], errors='coerce').dt.date

        # Remove rows with invalid dates
        df_ann_incStat   = df_ann_incStat[df_ann_incStat['fiscalDateEnding'].apply(lambda x: isinstance(x, datetime.date))]
        df_quar_incStat  = df_quar_incStat[df_quar_incStat['fiscalDateEnding'].apply(lambda x: isinstance(x, datetime.date))]
        df_ann_cashFlow  = df_ann_cashFlow[df_ann_cashFlow['fiscalDateEnding'].apply(lambda x: isinstance(x, datetime.date))]
        df_quar_cashFlow = df_quar_cashFlow[df_quar_cashFlow['fiscalDateEnding'].apply(lambda x: isinstance(x, datetime.date))]
        df_ann_balSheet  = df_ann_balSheet[df_ann_balSheet['fiscalDateEnding'].apply(lambda x: isinstance(x, datetime.date))]
        df_quar_balSheet = df_quar_balSheet[df_quar_balSheet['fiscalDateEnding'].apply(lambda x: isinstance(x, datetime.date))]
        df_ann_earnings  = df_ann_earnings[df_ann_earnings['fiscalDateEnding'].apply(lambda x: isinstance(x, datetime.date))]
        df_quar_earnings = df_quar_earnings[df_quar_earnings['fiscalDateEnding'].apply(lambda x: isinstance(x, datetime.date))]
        
        # Sort by fiscalDateEnding
        df_ann_incStat   = df_ann_incStat.sort_values(by='fiscalDateEnding').reset_index(drop=True).convert_dtypes()
        df_quar_incStat  = df_quar_incStat.sort_values(by='fiscalDateEnding').reset_index(drop=True).convert_dtypes()
        df_ann_cashFlow  = df_ann_cashFlow.sort_values(by='fiscalDateEnding').reset_index(drop=True).convert_dtypes()
        df_quar_cashFlow = df_quar_cashFlow.sort_values(by='fiscalDateEnding').reset_index(drop=True).convert_dtypes()
        df_ann_balSheet  = df_ann_balSheet.sort_values(by='fiscalDateEnding').reset_index(drop=True).convert_dtypes()
        df_quar_balSheet = df_quar_balSheet.sort_values(by='fiscalDateEnding').reset_index(drop=True).convert_dtypes()
        df_ann_earnings  = df_ann_earnings.sort_values(by='fiscalDateEnding').reset_index(drop=True).convert_dtypes()
        df_quar_earnings = df_quar_earnings.sort_values(by='fiscalDateEnding').reset_index(drop=True).convert_dtypes()

        # Cast to numeric
        exclude_cols = ['fiscalDateEnding','reportedDate', 'reportedCurrency', 'reportTime']
        df_ann_incStat   = df_ann_incStat.apply(lambda x:   pd.to_numeric(x, errors='coerce').fillna(pd.NA) if x.name not in exclude_cols else x)
        df_quar_incStat  = df_quar_incStat.apply(lambda x:  pd.to_numeric(x, errors='coerce').fillna(pd.NA) if x.name not in exclude_cols else x)
        df_ann_cashFlow  = df_ann_cashFlow.apply(lambda x:  pd.to_numeric(x, errors='coerce').fillna(pd.NA) if x.name not in exclude_cols else x)
        df_quar_cashFlow = df_quar_cashFlow.apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(pd.NA) if x.name not in exclude_cols else x)
        df_ann_balSheet  = df_ann_balSheet.apply(lambda x:  pd.to_numeric(x, errors='coerce').fillna(pd.NA) if x.name not in exclude_cols else x)
        df_quar_balSheet = df_quar_balSheet.apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(pd.NA) if x.name not in exclude_cols else x)
        df_ann_earnings  = df_ann_earnings.apply(lambda x:  pd.to_numeric(x, errors='coerce').fillna(pd.NA) if x.name not in exclude_cols else x)
        df_quar_earnings = df_quar_earnings.apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(pd.NA) if x.name not in exclude_cols else x)

        # Merge all financials
        financials_an = df_ann_earnings.merge(df_ann_incStat, on="fiscalDateEnding", how="outer")
        financials_an = financials_an.merge(df_ann_balSheet, on="fiscalDateEnding", how="outer")
        financials_an = financials_an.merge(df_ann_cashFlow, on="fiscalDateEnding", how="outer")
        
        financials_quar = df_quar_earnings.merge(df_quar_incStat, on="fiscalDateEnding", how="outer")
        financials_quar = financials_quar.merge(df_quar_balSheet, on="fiscalDateEnding", how="outer")
        financials_quar = financials_quar.merge(df_quar_cashFlow, on="fiscalDateEnding", how="outer")

        # Drop not assetdata columns
        finq_cols = ['fiscalDateEnding','reportedDate','reportedEPS','estimatedEPS',
                    'surprise','surprisePercentage','reportTime','grossProfit','totalRevenue',
                    'ebit','ebitda','totalAssets','totalCurrentLiabilities',
                    'totalShareholderEquity','commonStockSharesOutstanding','operatingCashflow']
        fina_cols = ['fiscalDateEnding','reportedEPS','grossProfit','totalRevenue',
                    'ebit','ebitda','totalAssets','totalCurrentLiabilities',
                    'totalShareholderEquity','operatingCashflow']
        financials_an = financials_an[fina_cols]
        financials_quar = financials_quar[finq_cols]

        # Pre-set dtypes
        financials_an = financials_an.convert_dtypes()
        financials_quar = financials_quar.convert_dtypes()
        financials_quar = financials_quar.astype({'reportedDate': 'string', 'reportTime': 'string'})

        # Clean data
        financials_quar = CleanData.financial_fiscalDateIncongruence(financials_quar, daysDiscrep = 15)
        financials_an = CleanData.financial_fiscalDateIncongruence(financials_an, daysDiscrep = 60)
        financials_an = CleanData.financial_lastRow_removeIfOutOfFiscal(financials_an)

        # Reset Index
        financials_an.reset_index(drop=True, inplace=True)
        financials_quar.reset_index(drop=True, inplace=True)

        # Special columns
        financials_quar['reportedDate'] = pd.to_datetime(financials_quar['reportedDate'], errors='coerce').dt.date.fillna(pd.NA)
        financials_quar['reportTime'] = financials_quar['reportTime'].where(financials_quar['reportTime'].isin(['pre-market','post-market', pd.NA]), pd.NA)

        # Convert date to str (as of definition AssetData)
        financials_an['fiscalDateEnding']   = financials_an['fiscalDateEnding'].apply(lambda ts: ts.strftime("%Y-%m-%d"))
        financials_quar['fiscalDateEnding'] = financials_quar['fiscalDateEnding'].apply(lambda ts: ts.strftime("%Y-%m-%d"))
        financials_quar['reportedDate']     = financials_quar['reportedDate'].apply(lambda ts: ts.strftime("%Y-%m-%d") if (ts is not None and ts is not pd.NA) else pd.NA)

        financials_an = financials_an.astype({
            'fiscalDateEnding'        : 'string',
            'reportedEPS'             : 'Float64',
            'grossProfit'             : 'Float64',
            'totalRevenue'            : 'Float64',
            'ebit'                    : 'Float64',
            'ebitda'                  : 'Float64',
            'totalAssets'             : 'Float64',
            'totalCurrentLiabilities' : 'Float64',
            'totalShareholderEquity'  : 'Float64',
            'operatingCashflow'       : 'Float64'
        })
        financials_quar = financials_quar.astype({
            'fiscalDateEnding'            : 'string',
            'reportedDate'                : 'string',
            'reportedEPS'                 : 'Float64',
            'estimatedEPS'                : 'Float64',
            'surprise'                    : 'Float64',
            'surprisePercentage'          : 'Float64',
            'reportTime'                  : 'string',
            'grossProfit'                 : 'Float64',
            'totalRevenue'                : 'Float64',
            'ebit'                        : 'Float64',
            'ebitda'                      : 'Float64',
            'totalAssets'                 : 'Float64',
            'totalCurrentLiabilities'     : 'Float64',
            'totalShareholderEquity'      : 'Float64',
            'commonStockSharesOutstanding': 'Float64',
            'operatingCashflow'           : 'Float64'
        })

        # Validate data
        self.validate_financials(financials_quar, financials_an)
        
        return financials_an, financials_quar
    
    def validate_shareprice(self, sData: pd.DataFrame) -> None:
        """
        Used for Manual checks on all incoming data to the Parser.
        Does not raise errors, but logs them.
        """
        if sData.empty:
            logger.error("PARSER VALIDATION: Shareprice data is empty.")
        
        # Check for negative values in price columns
        price_columns = ['Open', 'High', 'Low', 'Close', 'AdjClose']
        if (sData[price_columns] < 0).any().any():
            logger.error("PARSER VALIDATION: Float Shareprice data contains negative values.")

        # 3. shareprice
        if sData is not None:
            exp = ['Date','Open','High','Low','Close','AdjClose','Volume','Dividends','Splits']
            if list(sData.columns) != exp:
                logger.error(f"PARSER VALIDATION: Shareprice data columns must be {exp}")
            self.__check_date_col(sData, 'Date')
            for c in exp[1:]:
                self.__check_numeric_col(sData, c)
        
        return True
    
    def validate_financials(self, finquar: pd.DataFrame, finann: pd.DataFrame) -> None:
        """
        Used for Manual checks on all incoming data to the Parser.
        Does not raise errors, but logs them.
        """
        if finquar.empty:
            logger.error("PARSER VALIDATION: Finanicals quarterly is empty.")
        if finann.empty:
            logger.error("PARSER VALIDATION: Finanicals annual is empty.")
        
        # 4. financials_quarterly
        VALID_REPORT_TIMES = {"pre-market", "post-market", pd.NA}
        DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if finquar is not None:
            exp_q = [
                'fiscalDateEnding','reportedDate','reportedEPS','estimatedEPS','surprise',
                'surprisePercentage','reportTime','grossProfit','totalRevenue','ebit',
                'ebitda','totalAssets','totalCurrentLiabilities','totalShareholderEquity',
                'commonStockSharesOutstanding','operatingCashflow'
            ]
            if list(finquar.columns) != exp_q:
                logger.error(f"PARSER VALIDATION: financials_quarterly.columns must be {exp_q}")

            #fiscalDateEnding
            self.__check_date_col(finquar, 'fiscalDateEnding')

            # reportedDate
            rd = finquar['reportedDate']
            if rd.dtype != pd.StringDtype(storage="python"):
                logger.error(f"PARSER VALIDATION: 'reportedDate' must be dtype string[python], got {rd.dtype}")
            mask = rd.notna() & ~rd.astype(str).str.match(DATE_RE)
            bad  = finquar[mask]
            if not bad.empty:
                logger.error(f"PARSER VALIDATION: Column {'reportedDate'} has invalid date-strings: {bad['reportedDate'].unique().tolist()}")

            # reportTime
            rt = finquar['reportTime']
            if rt.dtype != pd.StringDtype(storage="python"):
                logger.error(f"PARSER VALIDATION: 'reportTime' must be dtype str, got {rt.dtype}")
            if not rt.empty and (rt.dtype != pd.StringDtype(storage="python") or not set(rt.unique()).issubset(VALID_REPORT_TIMES)):
                logger.error(f"PARSER VALIDATION: reportTime must be in {VALID_REPORT_TIMES}, got {rt.unique()}")
            
            # numeric
            for c in exp_q[2:]:
                if c not in {'reportTime'}:
                    self.__check_numeric_col(finquar, c)

        # 5. financials_annually
        if finann is not None:
            exp_a = [
                'fiscalDateEnding','reportedEPS','grossProfit','totalRevenue','ebit',
                'ebitda','totalAssets','totalCurrentLiabilities','totalShareholderEquity',
                'operatingCashflow'
            ]
            if list(finann.columns) != exp_a:
                logger.error(f"PARSER VALIDATION: financials_annually.columns must be {exp_a}")
            self.__check_date_col(finann, 'fiscalDateEnding')
            for c in exp_a[1:]:
                self.__check_numeric_col(finann, c)
        
        return True
    
    # Helper to check a date‐string column
    def __check_date_col(self, df: pd.DataFrame, col: str):
        DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        # accept object (py-str) or pandas StringDtype
        if not is_string_dtype(df[col]):
            logger.error(f"PARSER VALIDATION: Column {col} must be a string dtype, got {df[col].dtype}")

        # find any values which, when cast to str, don’t match YYYY-MM-DD
        bad = df[~df[col].astype(str).str.match(DATE_RE)]
        if not bad.empty:
            logger.error(f"PARSER VALIDATION: Column {col} has invalid date‐strings: {bad[col].unique().tolist()}")

    # Helper to check float columns
    def __check_numeric_col(self, df: pd.DataFrame, col: str):
        if not is_numeric_dtype(df[col]):
            logger.error(f"PARSER VALIDATION: Column {col} must be numeric dtype, got {df[col].dtype}")