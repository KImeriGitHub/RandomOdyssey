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
    
    num_quar_lag = 12   # number of lags to consider for quarterly data
    num_ann_lag = 3    # number of lags to consider for annual data
    
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'timesteps': 10,
    }
    
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
        'commonStockSharesOutstanding',
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
    ]
    
    catav_ann_lag = [
        'grossProfit',
        'ebit',
        'ebitda',
        'totalAssets',
        'totalCurrentLiabilities',
        'totalShareholderEquity',
        'operatingCashflow',
    ]
    
    catav_binary = [
        'reportTime', #0 for pre and 1 for post
    ]
    
    def __init__(self, asset: AssetDataPolars, 
            lagList: List[int] = [],
            params: dict = None):
        self.asset = asset
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        
        self.timesteps = self.params['timesteps']
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        
        self.fin_quar = self.asset.financials_quarterly
        self.fin_ann = self.asset.financials_annually
        self.shareprice = self.asset.shareprice.clone()
        
        #Columns to use at feature
        self.columns_toFeature_quar = []  # corresponds to columns in self.fin_quar
        self.columns_toFeature_ann = [] # corresponds to columns in self.fin_ann
        self.columns_toFeature_metric = [] # corresponds to columns in self.asset.shareprice
        self.columns_toFeature_timestep = [] # corresponds to columns in self.asset.shareprice
        
        # Make extra columns
        self.__operateOnFinData()
        self.__operateOnFinData_lag()
        self.__operateOnPriceData()
        self.__operateOnPriceData_lag(lagList)
        
        #make sure that some categories are in other categories
        assert all(item in self.catav_quar for item in self.catav_binary)
        assert all(item in self.catav_quar for item in self.catav_quar_lag)
        assert all(item in self.catav_ann for item in self.catav_ann_lag)
        
        assert all(item in self.fin_quar.columns for item in self.columns_toFeature_quar)
        assert all(item in self.fin_ann.columns for item in self.columns_toFeature_ann)
        assert all(item in self.shareprice.columns for item in self.columns_toFeature_metric)
        assert all(item in self.shareprice.columns for item in self.columns_toFeature_timestep)
        
    def __operateOnFinData(self):
        columns_notToDivide = [
            "fiscalDateEnding", 'reportedDate','totalRevenue', 
            'reportedEPS','estimatedEPS','surprise',
            'surprisePercentage','reportTime',
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
              .alias(f"{col}_nivRev")
            for col in columns_ann_toDivide
            if self.fin_ann.schema[col].is_numeric()
        ])
        self.columns_toFeature_ann += [
            f"{col}_nivRev" for col in columns_ann_toDivide if self.fin_ann.schema[col].is_numeric()
        ]
        
        self.fin_quar = self.fin_quar.with_columns([
            pl.when(pl.col("totalRevenue").is_not_null() & pl.col(col).is_not_null())
              .then(pl.col(col) / pl.col("totalRevenue"))
              .otherwise(None)
              .alias(f"{col}_nivRev")
            for col in columns_quar_toDivide
            if self.fin_quar.schema[col].is_numeric()
        ])
        self.columns_toFeature_quar += [
            f"{col}_nivRev" for col in columns_quar_toDivide if self.fin_quar.schema[col].is_numeric()
        ]
        
        # divide surprisePercentage by 1000
        if "surprisePercentage" in self.fin_quar.columns and self.fin_quar["surprisePercentage"].count() > 0:
            self.fin_quar = self.fin_quar.with_columns(
                (pl.col("surprisePercentage") / 1000.0).alias("surprisePercentage")
            )
            self.columns_toFeature_quar += ["surprisePercentage"]
        
        # keep totalRevenue as a ranking feature and scale down
        self.fin_ann = self.fin_ann.with_columns(
            (pl.col("totalRevenue").log().fill_nan(0.0) / 100.0).alias("totalRevenue_RANK")
        )
        self.columns_toFeature_ann += ["totalRevenue_RANK"]
        
        self.fin_quar = self.fin_quar.with_columns(
            (pl.col("totalRevenue").log().fill_nan(0.0) / 100.0).alias("totalRevenue_RANK")
        )
        self.columns_toFeature_quar += ["totalRevenue_RANK"]
        
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
        self.columns_toFeature_quar += ["reportTime"]
    
    def __operateOnFinData_lag(self):  
        for lag in range(1, self.num_quar_lag+1):
            self.fin_quar = self.fin_quar.with_columns([
                (pl.col(col).shift(lag) / pl.col("totalRevenue")).alias(f"{col}_nivRevLag_qm{lag}")
                for col in self.catav_quar_lag
            ]).with_columns([
                (pl.col(col) / pl.col(col).shift(lag)).alias(f"{col}_lagquot_qm{lag}")
                for col in self.catav_quar_lag
            ])
            self.columns_toFeature_quar += (
                [f"{col}_nivRevLag_qm{lag}" for col in self.catav_quar_lag] +
                [f"{col}_lagquot_qm{lag}" for col in self.catav_quar_lag]
            )
            
        for lag in range(1, self.num_ann_lag+1):
            self.fin_ann = self.fin_ann.with_columns([
                (pl.col(col).shift(lag) / pl.col("totalRevenue")).alias(f"{col}_nivRevLag_am{lag}")
                for col in self.catav_ann_lag
            ]).with_columns([
                (pl.col(col) / pl.col(col).shift(lag)).alias(f"{col}_lagquot_am{lag}")
                for col in self.catav_ann_lag
            ])
            self.columns_toFeature_ann += (
                [f"{col}_nivRevLag_am{lag}" for col in self.catav_ann_lag] +
                [f"{col}_lagquot_am{lag}" for col in self.catav_ann_lag]
            )
                
    def __operateOnPriceData(self):
        metricsColumns_quar = ["reportedEPS", "estimatedEPS", "reportedDate", "totalShareholderEquity", "commonStockSharesOutstanding", "ebit_lagquot_qm1", "totalShareholderEquity_lagquot_qm1"]
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
        if "q_idx" not in self.shareprice.columns:
            self.shareprice = (
                self.shareprice.join_asof(self.fin_quar_join, on="Date", strategy="backward")
            )
        if "a_idx" not in self.shareprice.columns:
            self.shareprice = (
                self.shareprice.join_asof(self.fin_ann_join, on="Date", strategy="backward")
            )
        
        self.shareprice = self.shareprice.with_columns([
            pl.when(pl.col("reportedEPS") <= 1e-5)
              .then(1e-5)
              .otherwise( (pl.col("Close") / (pl.col("reportedEPS")*4)).log() )
              .alias("log_trailing_pe_ratio"),

            pl.when(pl.col("estimatedEPS") <= 1e-5)
              .then(1e-5)
              .otherwise( (pl.col("Close") / (pl.col("estimatedEPS")*4)).log() )
              .alias("log_forward_pe_ratio"),
              
            (pl.col("Close") / (pl.col("totalShareholderEquity")/pl.col("commonStockSharesOutstanding")))
              .log() 
              .alias("log_pb_ratio"),
        ])
        
        self.shareprice = self.shareprice.with_columns([
            (pl.col("Close") * pl.col("ebit_lagquot_qm1")).alias("close2ebit_lagquot_qm1"),
            (pl.col("Close") * pl.col("totalShareholderEquity_lagquot_qm1")).alias("close2totalShareholderEquity_lagquot_qm1"),
        ])
        
        self.columns_toFeature_metric += [
            "log_trailing_pe_ratio",
            "log_forward_pe_ratio",
            "log_pb_ratio",
            "close2ebit_lagquot_qm1",
            "close2totalShareholderEquity_lagquot_qm1",
        ]
        self.columns_toFeature_timestep += [
            "log_trailing_pe_ratio",
            "log_forward_pe_ratio",
            "log_pb_ratio",
            "close2ebit_lagquot_qm1",
            "close2totalShareholderEquity_lagquot_qm1",
        ]
        
        # add the difference of days to the reportedDate (max 30 days)
        self.shareprice = self.shareprice.with_columns(
            (pl.col("reportedDate") - pl.col("Date")).dt.total_days().alias("daysToReport")
        )
        self.shareprice = self.shareprice.with_columns(
            (pl.col("daysToReport").clip(0,30)/30.0).alias("daysToReport")
        )
        
        self.columns_toFeature_metric += ["daysToReport"]
        self.columns_toFeature_timestep += ["daysToReport"]
        
    def __operateOnPriceData_lag(self, lagList: List[int] = []):
        #add lagged metrics
        columns_toLag = ["log_trailing_pe_ratio", "log_forward_pe_ratio", "log_pb_ratio", 'daysToReport']
        for lag in lagList:
            self.shareprice = self.shareprice.with_columns([
                pl.col(col).shift(lag).alias(f"{col}_lag_m{lag}")
                for col in columns_toLag
            ])
            self.shareprice = self.shareprice.with_columns([
                (pl.col(col)/pl.col(col).shift(lag)).alias(f"{col}_lagquot_m{lag}")
                for col in columns_toLag if col != 'daysToReport'
            ])
            self.columns_toFeature_metric += (
                [f"{col}_lag_m{lag}" for col in columns_toLag] +
                [f"{col}_lagquot_m{lag}" for col in columns_toLag if col != 'daysToReport']
            )
        
        # self.columns_toFeature_timestep does not include lagged data
    
    def getFeatureNames(self) -> list[str]:
        features_names = []
        features_names += ["FinData_quar_" + val for val in self.columns_toFeature_quar]
        features_names += ["FinData_ann_" + val for val in self.columns_toFeature_ann]
        features_names += ["FinData_metrics_" + val for val in self.columns_toFeature_metric]
        
        return features_names
    
    def getTimeFeatureNames(self) -> list[str]:
        features_names = []
        features_names += ["FinData_metrics_" + val for val in self.columns_toFeature_timestep]
        
        return features_names
    
    def apply(self, date: pd.Timestamp, scaleToNiveau: float, idx: int = None) -> np.ndarray:
        # Find shareprice index if idx not provided
        if idx is None:
            idx = DPl(self.shareprice).getNextLowerOrEqualIndex(date)
        
        # Get corresponding row indexes
        q_idx = self.shareprice["q_idx"][idx]
        a_idx = self.shareprice["a_idx"][idx]
        
        features_quar = list(self.fin_quar[self.columns_toFeature_quar].row(q_idx))
        features_ann = list(self.fin_ann[self.columns_toFeature_ann].row(a_idx))
        features_metrics = list(self.shareprice[self.columns_toFeature_metric].row(idx))
                
        features = np.array(
            features_quar +
            features_ann +
            features_metrics
        )
        
        features = np.nan_to_num(features.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        return (features * scaleToNiveau).astype(np.float32)
    
    def apply_timeseries(self, date: pd.Timestamp, idx: int = None) -> np.ndarray:
        if idx is None:
            idx = DPl(self.shareprice).getNextLowerOrEqualIndex(date)
        if idx - (self.num_quar_lag + 1.6) * 3 * self.idxLengthOneMonth < 0:
            raise ValueError("Lag is too far back.")
            
        coreLen = len(self.getTimeFeatureNames())
        featuresMat = np.zeros((self.timesteps, coreLen))
        
        timeseries_ival = (int)((self.num_quar_lag * 3 * self.idxLengthOneMonth + self.idxLengthOneMonth * 1.5) // (self.timesteps-1))
        
        niveau = self.shareprice["Close"].item(idx)
        
        colNames = self.columns_toFeature_timestep
        for ts in range(0, self.timesteps):
            idx_ts_sp = idx - ((self.timesteps - 1) - ts) * timeseries_ival
            
            valList = [self.shareprice[colNames[i]].item(idx_ts_sp) for i in range(len(colNames))]
            
            isnanList = [valList[i] is None or np.isnan(valList[i]) for i in range(len(colNames))]
            
            featuresMat[ts, 0] = np.tanh((valList[0] - np.log(20.0))/np.log(10.0))/2.0 + 0.5 if not isnanList[0] else 0  # 20 is a avg value, 10 good ival value for pe ratio
            featuresMat[ts, 1] = np.tanh((valList[1] - np.log(20.0))/np.log(10.0))/2.0 + 0.5 if not isnanList[1] else 0 # 20 is a avg value, 10 good ival value for pe ratios
            featuresMat[ts, 2] = np.tanh((valList[2] - np.log( 6.0))/np.log( 2.0))/2.0 + 0.5  if not isnanList[2] else 0 # 6 is a avg value, 2 good ival value for pe ratios
            featuresMat[ts, 3] = np.tanh(valList[3]/niveau - 1.0)/2.0 + 0.5 if not isnanList[3] else 0
            featuresMat[ts, 4] = np.tanh(valList[4]/niveau - 1.0)/2.0 + 0.5 if not isnanList[4] else 0
            featuresMat[ts, 5] = np.tanh(valList[5])  if not isnanList[5] else 0  # daysToReport; ought to be between 0 and 1 already
            
        return featuresMat.astype(np.float32)
        
    