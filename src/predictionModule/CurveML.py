import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List
import xgboost as xgb
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.CurveAnalysis import CurveAnalysis
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.predictionModule.IML import IML

class CurveML(IML):
    __idxLengthOneMonth = 21

    def __init__(self, assets: Dict[str, AssetDataPolars], trainStartDate: pd.Timestamp, trainEndDate: pd.Timestamp):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets
        self.__trainStartDate: pd.Timestamp = trainStartDate
        self.__trainEndDate: pd.Timestamp = trainEndDate

    def establishAssetIdx(self) -> Dict:
        # FOR FASTER RUN: Establish index in dataframe to start date
        assetdateIdx = {}
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice.select(pl.col("Date").last()).item() < self.__trainStartDate:
                raise ValueError(f"Asset {ticker} history not old enough or startDate ({self.__trainStartDate}) too far back. We stop.")
            assetdateIdx[ticker] = DPl(asset.adjClosePrice).getIndex(self.__trainStartDate, pd.Timedelta(days=0.7))

        return assetdateIdx

    def __getFeaturesFromPrice(pastPrices: np.array) -> list[float]:
        n = len(pastPrices)
        if n==1:
            raise ValueError("Input to getFeatureFromPrice are invalid.")
        if n==2:
            return pastPrices
        
        x = np.arange(n)
        fx0: float = pastPrices[0]
        fxend: float = pastPrices[n-1]
        yfit = fx0 + (fxend-fx0)*(x/(n-1))
        skewedPrices = pastPrices-yfit
        fourierInput = np.concatenate((skewedPrices,np.flipud(-skewedPrices[:n-1])))
        res_cos, res_sin = SeriesExpansion.getFourierConst(fourierInput)
        res_cos=res_cos.T
        res_sin=res_sin.T

        features = [fxend, (fxend-fx0)/(n-1)]
        for i in range(1, (len(res_cos)//2) + 1):
            if i < len(res_cos):
                features.append(res_cos[i])
            if i < len(res_sin):
                features.append(res_sin[i])

        return features

    def prepareData(self):
        X = []
        y = []
        assetdateIdx = self.establishAssetIdx()
        processedCounter=0
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice is None or not 'AdjClose' in asset.adjClosePrice.columns:
                continue

            print(f"Processing asset: {asset.ticker}.  Processed {processedCounter} out of {len(self.__assets)}.")
            processedCounter += 1

            pricesArray = asset.adjClosePrice['AdjClose']
            datesArray = asset.adjClosePrice['Date']
            dates = pd.date_range(self.__trainStartDate, self.__trainEndDate, freq='B', tz='UTC') # 'B' for business days
            for date in dates:
                if not abs(datesArray.item(assetdateIdx[ticker]) - date) <= pd.Timedelta(hours=18):
                    if datesArray.item(assetdateIdx[ticker]) < date:
                        assetdateIdx[ticker] = DPl(asset.adjClosePrice).getNextLowerIndex(date)+1
                    continue
                
                for m in [6,12,18,24]:
                    aidx = assetdateIdx[ticker]
                    if (aidx - m * self.__idxLengthOneMonth)<0:
                        continue
                    pastPrices = pricesArray.slice(aidx-m * self.__idxLengthOneMonth, m * self.__idxLengthOneMonth +1).to_numpy()
                    futurePrices = pricesArray.slice((aidx+1),5).to_numpy()
                    n = len(pastPrices)
                    x0 = pastPrices[0]
                    xend = pastPrices[-1]


                    fit_results_cos, fit_results_sin = SeriesExpansion.getFourierConst(pastPrices)
                    fit_results = np.concatenate((fit_results_cos.real,fit_results_sin.real), axis=1)
                    features = [m,
                                fit_results['Coefficients'][0],
                                fit_results['Coefficients'][1],
                                fit_results['Coefficients'][2],
                                fit_results['Coefficients'][3],
                                fit_results['R_Squared'],
                                fit_results['Variance']]
                    
                    if not np.isfinite(pastPrices).all() \
                        or not np.isfinite(futurePrices).all()\
                        or not np.isfinite(features).all():
                        continue

                    X.append(features)
                    y.append(futurePrices.values.tolist())
        self.X = np.array(X)
        self.y = np.array(y)

    def traintestModel(self) -> xgb.XGBRegressor:
        self.prepareData()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2)
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(((y_test - y_pred)**2).mean())
        print(f'Test RMSE per day: {rmse}')

    def predictNextPrices(self, priceArray, ticker: str, m: int = 1) -> np.ndarray:
        # priceArray is array_like
        fit_results = CurveAnalysis.thirdDegreeFit(priceArray, ticker)
        features = [m,
                    fit_results['Coefficients'][0],
                    fit_results['Coefficients'][1],
                    fit_results['Coefficients'][2],
                    fit_results['Coefficients'][3],
                    fit_results['R_Squared'],
                    fit_results['Variance']]
        features = np.array(features).reshape(1, -1)
        predicted_prices = self.model.predict(features)
        return predicted_prices[0]
