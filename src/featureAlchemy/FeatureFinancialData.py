import numpy as np
import pandas as pd
import polars as pl
from typing import List

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
    
    catav_quar_lag = [
        'reportedEPS',
        'estimatedEPS',
        'surprise',
        'surprisePercentage',
        'grossProfit',
        'ebit',
        'ebitda',
        'totalAssets',
        'totalCurrentLiabilities',
        'totalShareholderEquity',
        'operatingCashflow',
        'profitLoss',
    ]
    
    catav_ann_lag = [
        'grossProfit',
        'ebit',
        'ebitda',
        'totalAssets',
        'totalCurrentLiabilities',
        'totalShareholderEquity',
        'operatingCashflow',
        'profitLoss',
    ]
    
    catav_binary = [
        'reportTime', #0 for pre and 1 for post
    ]
    
    def __init__(self, asset: AssetDataPolars, lagList: List[int] = []):
        self.asset = asset
        
        self.fin_quar = self.asset.financials_quarterly
        self.fin_ann = self.asset.financials_annually
        
        self.num_quar_lag = 8   # number of lags to consider for quarterly data
        self.num_ann_lag = 2    # number of lags to consider for annual data
        
        self.__operateOnFinData()
        self.__operateOnFinData_lag()
        self.__operateOnPriceData()
        self.__operateOnPriceData_lag(lagList)
        self.__configureColumns(lagList)
        
        #make sure that some categories are in other categories
        assert all(item in self.catav_quar for item in self.catav_binary)
        assert all(item in self.catav_quar for item in self.catav_quar_lag)
        assert all(item in self.catav_ann for item in self.catav_ann_lag)
        
        assert all(item in self.fin_quar.columns for item in self.columns_toFeature_rank)
        assert all(item in self.fin_quar.columns for item in self.columns_toFeature_quar)
        assert all(item in self.fin_ann.columns for item in self.columns_toFeature_ann)
        assert all(item in self.asset.shareprice.columns for item in self.columns_toFeature_metrics)
        
    def __operateOnFinData(self):
        columns_notToDivide = [
            "fiscalDateEnding", 'reportedDate','totalRevenue', 
            'reportedEPS','estimatedEPS','surprise',
            'surprisePercentage','reportTime'
        ]
        columns_ann_toDivide = [c for c in self.catav_ann if c not in columns_notToDivide]
        columns_quar_toDivide = [c for c in self.catav_quar if c not in columns_notToDivide]
        
        # Drop duplicate years in fin_ann, keep first entry
        if "fiscalDateEnding" in self.fin_ann.columns:
            self.fin_ann = (
                self.fin_ann
                .with_columns(pl.col("fiscalDateEnding").dt.year().alias("year"))
                .unique(subset=["year"], keep="first")
                .drop("year")
                .sort("fiscalDateEnding")
            )
        
        # Scale numeric columns by totalRevenue in a single pass
        self.fin_ann = self.fin_ann.with_columns([
            pl.when(pl.col("totalRevenue").is_not_null() & pl.col(col).is_not_null())
              .then(pl.col(col) / pl.col("totalRevenue"))
              .otherwise(None)
              .alias(col)
            for col in columns_ann_toDivide
            if self.fin_ann.schema[col].is_numeric()
        ])
        
        self.fin_quar = self.fin_quar.with_columns([
            pl.when(pl.col("totalRevenue").is_not_null() & pl.col(col).is_not_null())
              .then(pl.col(col) / pl.col("totalRevenue"))
              .otherwise(None)
              .alias(col)
            for col in columns_quar_toDivide
            if self.fin_quar.schema[col].is_numeric()
        ])
        
        # divide surprisePercentage by 1000
        if "surprisePercentage" in self.fin_quar.columns:
            self.fin_quar = self.fin_quar.with_columns(
                (pl.col("surprisePercentage") / 1000.0).alias("surprisePercentage")
            )
        
        # keep totalRevenue as a ranking feature and scale down
        self.fin_ann = self.fin_ann.with_columns(
            (pl.col("totalRevenue").log().fill_nan(0.0) / 100.0).alias("totalRevenue_RANK")
        )
        self.fin_quar = self.fin_quar.with_columns(
            (pl.col("totalRevenue").log().fill_nan(0.0) / 100.0).alias("totalRevenue_RANK")
        )
        
        # Operate on catav_binary 
        unique_strings = (
            self.fin_quar
                .select(pl.col("reportTime"))
                .unique()
                .drop_nulls()
                .to_series()
                .to_list()
        )
        allowed_values = {"pre-market", "post-market"}
        if not all(string_name in allowed_values for string_name in unique_strings):
            print(f"Column reportTime contains values other than 'pre-market' and 'post-market'.")

        self.fin_quar = self.fin_quar.with_columns(
            pl.when(pl.col("reportTime").is_null())
              .then(None)
              .when(pl.col("reportTime") == "pre-market")
              .then(0).otherwise(1)
              .alias("reportTime")
        )
    
    def __operateOnFinData_lag(self):  
        for lag in range(1, self.num_quar_lag+1):
            self.fin_quar = self.fin_quar.with_columns([
                pl.col(col).shift(lag).alias(f"{col}_lag_qm{lag}")
                for col in self.catav_quar_lag
            ]).with_columns([
                (pl.col(col) / pl.col(col).shift(lag)).alias(f"{col}_lagquot_qm{lag}")
                for col in self.catav_quar_lag
            ])
            
        for lag in range(1, self.num_ann_lag+1):
            self.fin_ann = self.fin_ann.with_columns([
                pl.col(col).shift(lag).alias(f"{col}_lag_qm{lag}")
                for col in self.catav_ann_lag
            ]).with_columns([
                (pl.col(col) / pl.col(col).shift(lag)).alias(f"{col}_lagquot_qm{lag}")
                for col in self.catav_ann_lag
            ])
                
    def __operateOnPriceData(self):
        metricsColumns_quar = ["reportedEPS", "estimatedEPS", "reportedDate"]
        metricsColumns_ann = []
        
        # Configure joining dataframe
        self.fin_quar_join = (self.fin_quar
            .with_row_count("q_idx")
            .rename({"fiscalDateEnding": "Date"})
            .select(["Date", "q_idx"] + metricsColumns_quar)
        )

        self.fin_ann_join = (self.fin_ann
            .with_row_count("a_idx")
            .rename({"fiscalDateEnding": "Date"})
            .select(["Date", "a_idx"] + metricsColumns_ann)
        )

        # Asof join only the indices
        if "q_idx" not in self.asset.shareprice.columns:
            self.asset.shareprice = (
                self.asset.shareprice.join_asof(self.fin_quar_join, on="Date", strategy="backward")
            )
        if "a_idx" not in self.asset.shareprice.columns:
            self.asset.shareprice = (
                self.asset.shareprice.join_asof(self.fin_ann_join, on="Date", strategy="backward")
            )
        
        self.asset.shareprice = self.asset.shareprice.with_columns([
            pl.when(pl.col("reportedEPS") <= 1e-5)
              .then(1e-5)
              .otherwise( (pl.col("Close") / pl.col("reportedEPS")).log() )
              .alias("log_trailing_pe_ratio"),

            pl.when(pl.col("estimatedEPS") <= 1e-5)
              .then(1e-5)
              .otherwise( (pl.col("Close") / pl.col("estimatedEPS")).log() )
              .alias("log_forward_pe_ratio"),
        ])
        
        # add the difference of days to the reportedDate (max 30 days)
        self.asset.shareprice = self.asset.shareprice.with_columns(
            (pl.col("reportedDate") - pl.col("Date")).dt.total_days().alias("daysToReport")
        )
        self.asset.shareprice = self.asset.shareprice.with_columns(
            (pl.col("daysToReport").clip(0,30)/30.0).alias("daysToReport")
        )
        
    def __operateOnPriceData_lag(self, lagList: List[int] = []):
        #add lagged metrics
        columns_toLag = ["log_trailing_pe_ratio", "log_forward_pe_ratio", 'daysToReport']
        for lag in lagList:
            self.asset.shareprice = self.asset.shareprice.with_columns([
                pl.col(col).shift(lag).alias(f"{col}_lag_m{lag}")
                for col in columns_toLag
            ])
            self.asset.shareprice = self.asset.shareprice.with_columns([
                (pl.col(col)/pl.col(col).shift(lag)).alias(f"{col}_lagquot_m{lag}")
                for col in columns_toLag if col != 'daysToReport'
            ])
        
    def __configureColumns(self, lagList: List[int] = []):
        columns_notToAdd = ["fiscalDateEnding", 'reportedDate', 'totalRevenue']
        self.columns_toFeature_rank = ["totalRevenue_RANK"]
        self.columns_toFeature_quar = [c for c in self.catav_quar if c not in columns_notToAdd]
        self.columns_toFeature_ann = [c for c in self.catav_ann if c not in columns_notToAdd]
        
        for lag in range(1, self.num_quar_lag+1):
            self.columns_toFeature_quar += (
                [val + '_lag_qm' + str(lag) for val in self.catav_quar_lag] +
                [val + '_lagquot_qm' + str(lag) for val in self.catav_quar_lag]
            )
        for lag in range(1, self.num_ann_lag+1):
            self.columns_toFeature_ann += (
                [val + '_lag_qm' + str(lag) for val in self.catav_ann_lag] +
                [val + '_lagquot_qm' + str(lag) for val in self.catav_ann_lag]
            )
        
        self.columns_toFeature_metrics = ["log_trailing_pe_ratio", "log_forward_pe_ratio", 'daysToReport']
        for col in self.columns_toFeature_metrics.copy():
            for lag in lagList:
                self.columns_toFeature_metrics.append(f"{col}_lag_m{lag}")
                self.columns_toFeature_metrics.append(f"{col}_lagquot_m{lag}") if col != 'daysToReport' else None
    
    def getFeatureNames(self) -> list[str]:
        features_names = []
        features_names += ["FinData_rank_" + val for val in self.columns_toFeature_rank]
        features_names += ["FinData_quar_" + val for val in self.columns_toFeature_quar]
        features_names += ["FinData_ann_" + val for val in self.columns_toFeature_ann]
        features_names += ["FinData_metrics_" + val for val in self.columns_toFeature_metrics]
        
        return features_names
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        """
        Retrieve a feature vector for a given date and multiply by `scaleToNiveau`.
        """
        # Find shareprice index if idx not provided
        if idx is None:
            idx = DPl(self.asset.shareprice).getNextLowerOrEqualIndex(date)
        
        # Get corresponding row indexes
        q_idx = self.asset.shareprice["q_idx"][idx]
        a_idx = self.asset.shareprice["a_idx"][idx]
        
        features_rank = list(self.fin_quar[self.columns_toFeature_rank].row(q_idx))
        features_quar = list(self.fin_quar[self.columns_toFeature_quar].row(q_idx))
        features_ann = list(self.fin_ann[self.columns_toFeature_ann].row(a_idx))
        features_metrics = list(self.asset.shareprice[self.columns_toFeature_metrics].row(idx))
                
        features = np.array(
            features_rank +
            features_quar +
            features_ann +
            features_metrics
        )
        
        features = np.nan_to_num(features.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        return features * scaleToNiveau