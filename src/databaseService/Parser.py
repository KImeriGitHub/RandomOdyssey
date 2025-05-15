from typing import Dict
import pandas as pd
import re
from typing import Dict
from pandas.api.types import is_float_dtype

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
            # build a mask: keep rows where *all* cols are notna, or it’s the last row
            last_idx = fullSharePrice.index[-1]
            mask = fullSharePrice[cols].notna().all(axis=1) | (fullSharePrice.index == last_idx)
            fullSharePrice = fullSharePrice[mask]
            
        # Convert date to str (as of definition AssetData)
        fullSharePrice['Date'] = fullSharePrice['Date'].apply(lambda ts: ts.strftime("%Y-%m-%d"))
        
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
        
        df_ann_incStat = pd.DataFrame(self.incStatementData['annualReports'])
        df_quar_incStat = pd.DataFrame(self.incStatementData['quarterlyReports'])
        
        df_ann_cashFlow = pd.DataFrame(self.cashFlowData['annualReports'])
        df_quar_cashFlow = pd.DataFrame(self.cashFlowData['quarterlyReports'])
        
        df_ann_balSheet = pd.DataFrame(self.balanceSheetData['annualReports'])
        df_quar_balSheet = pd.DataFrame(self.balanceSheetData['quarterlyReports'])
        
        df_ann_earnings = pd.DataFrame(self.earningsData['annualEarnings'])
        df_quar_earnings = pd.DataFrame(self.earningsData['quarterlyEarnings'])
        
        exclude_cols = ['fiscalDateEnding','reportedDate', 'reportedCurrency', 'reportTime']
        df_ann_incStat = df_ann_incStat.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
        df_quar_incStat = df_quar_incStat.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
        df_ann_cashFlow = df_ann_cashFlow.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
        df_quar_cashFlow = df_quar_cashFlow.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
        df_ann_balSheet = df_ann_balSheet.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
        df_quar_balSheet = df_quar_balSheet.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
        df_ann_earnings = df_ann_earnings.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
        df_quar_earnings = df_quar_earnings.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)

        df_ann_incStat.replace('None', pd.NA, inplace=True)
        df_quar_incStat.replace('None', pd.NA, inplace=True)
        df_ann_cashFlow.replace('None', pd.NA, inplace=True)
        df_quar_cashFlow.replace('None', pd.NA, inplace=True)
        df_ann_balSheet.replace('None', pd.NA, inplace=True)
        df_quar_balSheet.replace('None', pd.NA, inplace=True)
        df_ann_earnings.replace('None', pd.NA, inplace=True)
        df_quar_earnings.replace('None', pd.NA, inplace=True)

        df_ann_incStat['fiscalDateEnding'] = pd.to_datetime(df_ann_incStat['fiscalDateEnding'], utc=True)
        df_ann_incStat = df_ann_incStat.sort_values(by='fiscalDateEnding')
        df_quar_incStat['fiscalDateEnding'] = pd.to_datetime(df_quar_incStat['fiscalDateEnding'], utc=True)
        df_quar_incStat = df_quar_incStat.sort_values(by='fiscalDateEnding')
        df_ann_cashFlow['fiscalDateEnding'] = pd.to_datetime(df_ann_cashFlow['fiscalDateEnding'], utc=True)
        df_ann_cashFlow = df_ann_cashFlow.sort_values(by='fiscalDateEnding')
        df_quar_cashFlow['fiscalDateEnding'] = pd.to_datetime(df_quar_cashFlow['fiscalDateEnding'], utc=True)
        df_quar_cashFlow = df_quar_cashFlow.sort_values(by='fiscalDateEnding')
        df_ann_balSheet['fiscalDateEnding'] = pd.to_datetime(df_ann_balSheet['fiscalDateEnding'], utc=True)
        df_ann_balSheet = df_ann_balSheet.sort_values(by='fiscalDateEnding')
        df_quar_balSheet['fiscalDateEnding'] = pd.to_datetime(df_quar_balSheet['fiscalDateEnding'], utc=True)
        df_quar_balSheet = df_quar_balSheet.sort_values(by='fiscalDateEnding')
        df_ann_earnings['fiscalDateEnding'] = pd.to_datetime(df_ann_earnings['fiscalDateEnding'], utc=True)
        df_ann_earnings = df_ann_earnings.sort_values(by='fiscalDateEnding')
        df_quar_earnings['fiscalDateEnding'] = pd.to_datetime(df_quar_earnings['fiscalDateEnding'], utc=True)
        df_quar_earnings = df_quar_earnings.sort_values(by='fiscalDateEnding')

        df_quar_earnings['reportedDate'] = pd.to_datetime(df_quar_earnings['reportedDate'], utc=True)
        
        financials_an = pd.merge(df_ann_earnings, df_ann_incStat, on="fiscalDateEnding", how="outer")
        financials_an = financials_an.merge(df_ann_balSheet, on="fiscalDateEnding", how="outer")
        financials_an = financials_an.merge(df_ann_cashFlow, on="fiscalDateEnding", how="outer")
        
        financials_quar = pd.merge(df_quar_earnings, df_quar_incStat, on="fiscalDateEnding", how="outer")
        financials_quar = financials_quar.merge(df_quar_balSheet, on="fiscalDateEnding", how="outer")
        financials_quar = financials_quar.merge(df_quar_cashFlow, on="fiscalDateEnding", how="outer")
        
        financials_quar = CleanData.financial_fiscalDateIncongruence(financials_quar, daysDiscrep = 15)
        financials_an = CleanData.financial_fiscalDateIncongruence(financials_an, daysDiscrep = 60)
        financials_an = CleanData.financial_lastRow_removeIfOutOfFiscal(financials_an)
        
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
                self.__check_float_col(sData, c)
        
        return True
    
    # Helper to check a date‐string column
    def __check_date_col(self, df, col):
        DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if df[col].dtype != object:
            logger.error(f"PARSER VALIDATION: Shareprice data {col} must be dtype object (str), got {df[col].dtype}")
        bad = df[~df[col].astype(str).str.match(DATE_RE)]
        if not bad.empty:
            logger.error(f"PARSER VALIDATION: Shareprice data Column {col} has invalid dates:\n{bad[col].unique()}")
    # Helper to check float columns
    def __check_float_col(self, df, col):
        if not is_float_dtype(df[col]):
            logger.error(f"PARSER VALIDATION: Shareprice data {col} must be float dtype, got {df[col].dtype}")