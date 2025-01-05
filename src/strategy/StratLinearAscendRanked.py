from src.mathTools.CurveAnalysis import CurveAnalysis
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
from src.common.AssetDataPolars import AssetDataPolars
from src.common.Portfolio import Portfolio
from src.common.ActionCost import ActionCost
from typing import Dict, List
import pandas as pd
import polars as pl
import numpy as np

class StratLinearAscendRanked(IStrategy):
    __stoplossRatio = 0.92

    def __init__(self,
                 portfolio: Portfolio = None,
                 num_months: int = 2, 
                 num_choices: int = 1,
                 ):
        self.num_months: int = num_months
        self.num_choices: int = num_choices

        self.__assets: Dict[str, AssetDataPolars] = {}
        self.__portfolio: Portfolio = portfolio

        self.__stoplossLimit: Dict[str, float] = {}
        self.__blacklist: Dict[str, pd.Timestamp] = {}

        self.__assetdateIdx: Dict[str, int] = {}

    def sellOrders(self) -> Dict:
        sellOrders = {}
        for boughtTicker in self.__portfolio.positions.keys():
            asset: AssetDataPolars = self.__assets[boughtTicker]
            price: float = asset.adjClosePrice['AdjClose'].item(self.__assetdateIdx[boughtTicker])
            if price <= self.__stoplossLimit[boughtTicker]:
                sellOrders[boughtTicker] = {
                    "quantity": self.__portfolio.positions[boughtTicker],
                    "price": price
                }
        return sellOrders

    def updateStoplossLimit(self, buyorders: dict[str,]={}, sellorders: dict[str,]={}):
        # Deal with buy order
        for portticker in buyorders.keys():
            if not portticker in self.__stoplossLimit:
                self.__stoplossLimit[portticker] = self.__stoplossRatio
                continue

            asset: AssetDataPolars = self.__assets[portticker]
            price_data: float = asset.adjClosePrice["AdjClose"].item(self.__assetdateIdx[portticker])
            if price_data * self.__stoplossRatio > self.__stoplossLimit[portticker]:
                self.__stoplossLimit[portticker] =  price_data * self.__stoplossRatio

        # Deal with sell order
        for portticker in sellorders.keys():
            if not portticker in self.__stoplossLimit:
                continue

            del self.__stoplossLimit[portticker]

        # Update Stop loss
        for portticker in self.__stoplossLimit.keys():
            asset: AssetDataPolars = self.__assets[portticker]
            price_data: float = asset.adjClosePrice["AdjClose"].item(self.__assetdateIdx[portticker])
            if price_data * self.__stoplossRatio > self.__stoplossLimit[portticker]:
                self.__stoplossLimit[portticker] =  price_data * self.__stoplossRatio


    def preAnalyze(self) -> pl.DataFrame:
        modAssetList: List[str] = []
        priceData: float = 0.0
        for ticker in self.__assets.keys():
            asset: AssetDataPolars = self.__assets[ticker]
            priceData = asset.adjClosePrice['AdjClose'].item(self.__assetdateIdx[ticker])
            if priceData > 10.0:
                modAssetList.append(ticker)

        analysis_results = []
        startDateIdxDiff = self.num_months*21
        for ticker in modAssetList:
            asset: AssetDataPolars = self.__assets[ticker]
            priceData: pl.DataFrame = asset.adjClosePrice["AdjClose"].slice(self.__assetdateIdx[ticker]-startDateIdxDiff, startDateIdxDiff+1)

            # Store results
            analysis_results.append(CurveAnalysis.lineFit(priceData.to_numpy(), ticker))

        if not analysis_results:
            print("No assets available for analysis.")
            return
        
        return pl.DataFrame(analysis_results)

    def rankAssets(self) -> pl.DataFrame:
        results_df: pl.DataFrame = self.preAnalyze()

        # Step 1: Calculate the  quantile of the 'Slope' column
        quant = results_df.select(pl.col("Slope").quantile(0.90)).item()

        # Step 2: Assign 'Rankslope'
        # - Rank 1 if 'Slope' > quant
        # - Otherwise, assign a dense rank in descending order starting from 2
        results_df = results_df.with_columns([
            # Create a temporary column 'Slope_filtered' where 'Slope' <= quant
            pl.when(pl.col("Slope") <= quant)
              .then(pl.col("Slope"))
              .otherwise(None)
              .alias("Slope_filtered")
        ]).with_columns([
            # Compute dense rank for 'Slope_filtered' in descending order
            pl.col("Slope_filtered")
              .rank(method="dense", descending=True)
              .alias("Slope_rank")
        ]).with_columns([
            # Assign 'Rankslope': 1 if 'Slope' > quant, else 'Slope_rank' + 1
            pl.when(pl.col("Slope") > quant)
              .then(1)
              .otherwise(pl.col("Slope_rank") + 1)
              .alias("Rankslope")
        ]).drop(["Slope_filtered", "Slope_rank"])  # Remove temporary column

        # Step 3: Rank the 'Variance' column in ascending order
        results_df = results_df.with_columns([
            pl.col("Variance")
              .rank(method="ordinal", descending=False)
              .alias("Rankvar")
        ])
        # Step 4: Compute the 'Score' by summing 'Rankslope' and 'Rankvar'
        results_df = results_df.with_columns([
            (pl.col("Rankslope") + pl.col("Rankvar")).alias("Score")
        ])
        # Step 5: Sort the DataFrame by 'Score' in ascending order
        results_df = results_df.sort("Score", descending=False)

        # Step 6: Select the top choices based on available slots
        available_choices = self.num_choices - len(self.__portfolio.positions.keys())
        top_choices = results_df.head(available_choices)

        return top_choices
    
    def buyOrders(self) -> Dict:
        buyOrders = {}

        # Select top choices
        top_choices: pl.DataFrame = self.rankAssets()

        # Divide cash equally among selected stocks
        cashPerStock = self.__portfolio.cash / len(top_choices)

        for topTicker in top_choices["Ticker"].to_list():
            asset: AssetDataPolars = self.__assets[topTicker]
            price: float = asset.adjClosePrice["AdjClose"].item(self.__assetdateIdx[topTicker])
            quantity = np.floor((cashPerStock) / (price + ActionCost().buy(price)))
            if abs(quantity) > 0.00001:
                buyOrders[topTicker] = {
                    "quantity": quantity,
                    "price": price
                }

        return buyOrders

    def apply(self,
              assets: Dict[str, AssetDataPolars], 
              currentDate: pd.Timestamp, 
              assetdateIdx: Dict[str, int] = {}):
        
        self.__assets = assets
        self.__currentDate = currentDate
        self.__assetdateIdx = assetdateIdx

        sellOrders = self.sellOrders()
        self.sell(sellOrders, self.__portfolio, currentDate)

        self.updateStoplossLimit(sellorders = sellOrders)

        if not self.__portfolio.valueOverTime == [] \
            and self.__portfolio.cash/self.__portfolio.valueOverTime[-1][1] < self.__cashthreshhold \
            and not sellOrders:
            return  # Do not buy if positions are not empty and no assets were sold.

        buyOrders = self.buyOrders()
        self.buy(buyOrders, self.__portfolio, currentDate)

        self.updateStoplossLimit(buyorders = buyOrders)