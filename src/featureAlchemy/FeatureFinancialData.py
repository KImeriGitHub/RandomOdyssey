import numpy as np
import pandas as pd
import polars as pl
from typing import Dict

from src.common.AssetDataPolars import AssetDataPolars
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl

class FeatureFinancialData():
    """
    NOTE: Only alpha vantage implemented as of yet. 
    """
    
    operator = "alphavantage"
    
    # Class-level default parameters
    catav_quar = [
        'fiscalDateEnding',
        'reportedDate',
        'reportedEPS',
        'estimatedEPS',
        'surprise',
        'surprisePercentage',
        'reportTime',
        'grossProfit',
        'totalRevenue',
        'costOfRevenue',
        'operatingIncome',
        'sellingGeneralAndAdministrative',
        'operatingExpenses',
        'interestExpense',
        'ebit',
        'ebitda',
        'totalAssets',
        'totalCurrentAssets',
        'totalNonCurrentAssets',
        'shortTermInvestments',
        'totalCurrentLiabilities',
        'shortTermDebt',
        'totalShareholderEquity',
        'operatingCashflow',
        'changeInOperatingLiabilities',
        'profitLoss',
        'cashflowFromInvestment',
    ]
    
    catav_ann = [
        'fiscalDateEnding',
        'reportedDate',
        'reportedEPS',
        'grossProfit',
        'totalRevenue',
        'costOfRevenue',
        'operatingIncome',
        'sellingGeneralAndAdministrative',
        'operatingExpenses',
        'interestExpense',
        'ebit',
        'ebitda',
        'totalAssets',
        'totalCurrentAssets',
        'totalNonCurrentAssets',
        'shortTermInvestments',
        'totalCurrentLiabilities',
        'shortTermDebt',
        'totalShareholderEquity',
        'operatingCashflow',
        'changeInOperatingLiabilities',
        'profitLoss',
        'cashflowFromInvestment',
    ]
    
    catav_binary = [
        'reportTime', #0 for pre and 1 for post
    ]
    
    def __init__(self, asset: AssetDataPolars):
        self.asset = asset
        
        self.cat = self.cat_alphavantage if self.operator == "alphavantage" else self.cat_yfinance
        
        self.fin_quar = self.asset.financials_quarterly
        self.fin_ann = self.asset.financials_annually
        
        self.__operateOnFinData()
        self.__operateOnPriceData()
        
        # NOTE: divide surpricePercentage by 100
        # NOTE: fill in None strings with 0
        # Note: PE RATIO
        
    def __operateOnFinData(self):
        columns_notToDivide = ["fiscalDateEnding", 'reportedDate','totalRevenue', 'reportedEPS','estimatedEPS','surprise','surprisePercentage','reportTime']
        columns_ann_toDivide = [item for item in self.catav_ann if item not in columns_notToDivide]
        columns_quar_toDivide = [item for item in self.catav_quar if item not in columns_notToDivide]
        
        # Divide all numeric columns by "totalRevenue", handling missing values safely
        self.fin_ann = self.fin_ann.with_columns(
            [
                (pl.col(col) / pl.col("totalRevenue")).alias(col)
                for col in columns_ann_toDivide
                if self.fin_ann[col].dtype.is_numeric() and col != "totalRevenue"
            ]
        )
        
        self.fin_quar = self.fin_quar.with_columns(
            [
                (pl.col(col) / pl.col("totalRevenue")).alias(col)
                for col in columns_quar_toDivide
                if self.fin_quar[col].dtype.is_numeric() and col != "totalRevenue"
            ]
        )
        
        # divide surprisePercentage by 1000
        self.fin_quar = self.fin_quar.with_columns(
                (pl.col("surprisePercentage") / 1000.0).alias("surprisePercentage")
        )
        
        # Convert catav_binary from binary string to 0 and 1
        for col in self.catav_binary:
            # Get unique values in the column nan values are discarded
            unique_values = self.fin_quar[col].unique().drop_nulls().to_list() 
            if len(unique_values) == 2:
                self.fin_quar = self.fin_quar.with_columns(
                    pl.when(pl.col(col).is_null()).then(None).  # Preserve NaN values
                    when(pl.col(col) == unique_values[0]).then(0).otherwise(1).alias(col)
                )
                
        # If there are two entries in fiscalDateEnding in fin_ann that are in the same year, keep the first one
        self.fin_ann = (
            self.fin_ann.with_columns(pl.col("fiscalDateEnding").dt.year().alias("year"))
            .unique(subset=["year"], keep="first")
            .drop("year")
            .sort("fiscalDateEnding")
        )
            
                
    def __operateOnPriceData(self):
        # Add row indices to the fin_quar and fin_ann tables
        fin_quar2 = (self.fin_quar
            .with_row_count("q_idx")
            .rename({"fiscalDateEnding": "Date"})
            .select(["Date", "q_idx"])
        )

        fin_ann2 = (self.fin_ann
            .with_row_count("a_idx")
            .rename({"fiscalDateEnding": "Date"})
            .select(["Date", "a_idx"])
        )

        # Asof join only the indices
        self.asset.shareprice = (
            self.asset.shareprice
            .join_asof(fin_quar2, on="Date", strategy="backward")
            .join_asof(fin_ann2, on="Date", strategy="backward")
        )
        
    def __extendToFinancialMetrics(self):
        # Extend the shareprice table with the financial data
        self.finMetric = self.asset.shareprice.select("Date")
        
    
    def getFeatureNames(self) -> list[str]:
        features_names = ["FinData_" + val for val in self.cat]
            
        #TODO
        
        return features_names
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = -1):
        if idx<0:
            idx = DPl(self.asset.adjClosePrice).getNextLowerIndex(date)+1
        
        #Todo
        
        # Create a one-hot encoding where the category matches the sector
        features = np.array([1 if category == sector else 0 for category in self.cat])
        
        return features*scaleToNiveau