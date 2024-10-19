import numpy as np
import pandas as pd
from typing import Dict
import xgboost as xgb
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.common.AssetData import AssetData
from src.mathTools.CurveAnalysis import CurveAnalysis
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPandas as DFTO
from src.predictionModule.IML import IML

class CurveML(IML):
    __idxLengthOneMonth = 21

    def __init__(self, assets: Dict[str, AssetData], trainStartDate: pd.Timestamp, trainEndDate: pd.Timestamp):
        super().__init__()
        self.__assets = assets
        self.__trainStartDate = trainStartDate
        self.__trainEndDate = trainEndDate

    def establishAssetIdx(self) -> Dict:
        # FOR FASTER RUN: Establish index in dataframe to start date
        assetdateIdx = {}
        for ticker, asset in self.__assets.items():
            if asset.shareprice.index[-1] < self.__trainStartDate:
                raise ValueError(f"Asset {ticker} history not old enough or startDate ({self.startDate}) too far back. We stop.")

            assetdateIdx[ticker] = DFTO(asset.shareprice).getNextLowerIndex(self.__trainStartDate)+1

        return assetdateIdx

    def prepareData(self):
        X = []
        y = []
        assetdateIdx = self.establishAssetIdx()
        processedCounter=0
        for ticker, asset in self.__assets.items():
            if asset.shareprice is None or not 'Close' in asset.shareprice.columns:
                continue

            print(f"Processing asset: {asset.ticker}.  Processed {processedCounter} out of {len(self.__assets)}.")
            processedCounter += 1

            pricesArray = asset.shareprice['Close']
            pricesArray = pricesArray.resample('B').mean().dropna()
            dates = pd.date_range(self.__trainStartDate, self.__trainEndDate, freq='B', tz='UTC') # 'B' for business days
            for date in dates:
                if not abs(pricesArray.index[assetdateIdx[ticker]] - date) <= pd.Timedelta(hours=18):
                    if pricesArray.index[assetdateIdx[ticker]] < date:
                        assetdateIdx[ticker] = DFTO(asset.shareprice).getNextLowerIndex(date)+1
                    continue
                
                for m in [1,2,3,4,5,6]:
                    aidx = assetdateIdx[ticker]
                    if (aidx - m * self.__idxLengthOneMonth)<0:
                        continue
                    pastPrices = pricesArray.iloc[(aidx - m * self.__idxLengthOneMonth) : (aidx+1)]
                    futurePrices = pricesArray.iloc[(aidx+1):(aidx+6)]
                    fit_results = CurveAnalysis.thirdDegreeFit(pastPrices, ticker)
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
