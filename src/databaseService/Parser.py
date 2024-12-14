from typing import Dict
import pandas as pd

# For Alpha Vantage
class Parser_AV():
    def __init__(self, incStatementData: Dict, cashFlowData: Dict, balanceSheetData: Dict, earningsData: Dict):
        self.incStatementData = incStatementData
        self.cashFlowData = cashFlowData
        self.balanceSheetData = balanceSheetData
        self.earningsData = earningsData
    
    def to_pandas(self):
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
        
        return financials_an, financials_quar