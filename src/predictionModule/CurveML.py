import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List
import xgboost as xgb
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.CurveAnalysis import CurveAnalysis
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.predictionModule.IML import IML

class CurveML(IML):
    __idxLengthOneMonth = 21

    def __init__(self, assets: Dict[str, AssetDataPolars], trainStartDate: pd.Timestamp, trainEndDate: pd.Timestamp, numOfMonths: int = 24):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets
        self.__trainStartDate: pd.Timestamp = trainStartDate
        self.__trainEndDate: pd.Timestamp = trainEndDate

        self._numOfMonths = numOfMonths
        self.X = []
        self.y = []
        self.__dataIsPrepared = False

    def establishAssetIdx(self) -> Dict:
        # FOR FASTER RUN: Establish index in dataframe to start date
        assetdateIdx = {}
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice["Date"].item(-1) < self.__trainStartDate:
                raise ValueError(f"Asset {ticker} history not old enough or startDate ({self.__trainStartDate}) too far back. We stop.")
            assetdateIdx[ticker] = DPl(asset.adjClosePrice).getIndex(self.__trainStartDate, pd.Timedelta(days=0.7))

        return assetdateIdx

    @staticmethod
    def __getFeaturesFromPrice(pastPrices: np.array, includeLastPrice = False, multFactor: int = 8, fouriercutoff: int = 100) -> list[float]:
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
        cs = CubicSpline(np.arange(len(fourierInput)), fourierInput)
        fourierInputSpline = cs(np.linspace(0, len(fourierInput)-1, 1 + (len(fourierInput) - 1) * multFactor))
        fourierInputSmooth = gaussian_filter1d(fourierInputSpline, sigma=np.max([len(fourierInputSpline)//((multFactor-1)*n),1]))
        res_cos, res_sin = SeriesExpansion.getFourierConst(fourierInputSmooth)
        res_cos=res_cos.T.real.flatten()
        res_sin=res_sin.T.real.flatten()

        if len(res_cos) < fouriercutoff:
            raise Warning("fouriercutoff is bigger than the array itself.")

        features = []
        if includeLastPrice:
            features.append(fxend)
            features.append((fxend-fx0)/(n-1))
        else:
            features.append((1-fx0/fxend)/(n-1))

        for i in range(0, (len(res_cos)//2) + 1):
            if i > fouriercutoff//2:
                break

            if i < len(res_cos):
                features.append(res_cos[i])
            if i < len(res_sin):
                features.append(res_sin[i])

        return features
    
    @staticmethod
    def __getTargetFromPrice(futurePrices: np.array, percInt: list[float]) -> list[float]:
        """
        Let A be a entry in futurePrices.
        If A is between 0 and the first entry of percInt the method returns on first result entry 0. 
        If A is between the first entry of percInt and the second entry of percInt it returns on second result entry 1, and so on.
        If A is negative it does the above for -A and returns the negativ result entry.
        """
        res = np.zeros(len(futurePrices))
        for j, A in enumerate(futurePrices):
            if A == 0:
                res[j] = 0
                continue

            sign = 1 if A > 0 else -1
            A_abs = abs(A)
            if A_abs <= percInt[0]:
                res[j] = 0
                continue
            if A_abs > percInt[-1]:
                res[j] = len(percInt) * sign
                continue
            for i in range(1, len(percInt)):
                if percInt[i - 1] < A_abs <= percInt[i]:
                    res[j] = i * sign

        return res.tolist()

    def prepareData(self):
        X = []
        y = []
        lenClassInterval = 10
        #classificationIntervals = [0.01*(2*i+1) for i in range(0,lenClassInterval)]
        classificationIntervals = np.exp(np.linspace(start=np.log(1.001),stop=np.log(1.15), num=lenClassInterval).tolist())-1
        assetdateIdx = self.establishAssetIdx()
        processedCounter=0
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice is None or not 'AdjClose' in asset.adjClosePrice.columns:
                continue

            print(f"Processing asset: {asset.ticker}.  Processed {processedCounter} out of {len(self.__assets)}.")
            processedCounter += 1

            pricesArray = asset.adjClosePrice['AdjClose']
            datesArray = asset.adjClosePrice['Date']
            dates = pd.date_range(self.__trainStartDate, self.__trainEndDate, freq='B') # 'B' for business days
            spare_dates = pd.DatetimeIndex(np.random.choice(dates, size=len(dates)//6, replace=False))
            for date in spare_dates:
                if not abs(datesArray.item(assetdateIdx[ticker]) - date) <= pd.Timedelta(hours=18):
                    if datesArray.item(assetdateIdx[ticker]) < date:
                        assetdateIdx[ticker] = DPl(asset.adjClosePrice).getNextLowerIndex(date)+1
                
                m = self._numOfMonths
                aidx = assetdateIdx[ticker]
                if (aidx - m * self.__idxLengthOneMonth)<0:
                    continue
                pastPrices = pricesArray.slice(aidx-m * self.__idxLengthOneMonth, m * self.__idxLengthOneMonth +1).to_numpy()
                futurePrices = pricesArray.slice((aidx+1),1).to_numpy()
                
                features = self.__getFeaturesFromPrice(pastPrices/pastPrices[-1], multFactor=16)
                target = self.__getTargetFromPrice(futurePrices/pastPrices[-1]-1, classificationIntervals)
                
                if not np.isfinite(pastPrices).all() \
                    or not np.isfinite(futurePrices).all()\
                    or not np.isfinite(features).all():
                    raise ValueError("Stock Prices are not complete")
                X.append(features)
                y.append(target)

        self.__dataIsPrepared = True
        self.X = np.array(X)
        self.y = np.round(np.array(y)).astype(int)+lenClassInterval

    def traintestXGBoostModel(self) -> xgb.XGBRegressor:
        raise NotImplementedError("XGBoost is not implemented yet.")
    
        #self.prepareData()
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2)
        #self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        #self.model.fit(X_train, y_train)
        #y_pred = self.model.predict(X_test)
        #rmse = np.sqrt(((y_test - y_pred)**2).mean())
        #print(f'Test RMSE per day: {rmse}')

    def traintestCNNModel(self) -> xgb.XGBRegressor:
        if not self.__dataIsPrepared:
            self.prepareData()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2)
        # Binarize the labels
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(y_train)
        y_test = mlb.transform(y_test)

        input_shape = X_train.shape[1]

        # Define the model
        self.CNNModel = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(mlb.classes_), activation='sigmoid')
        ])

        # Compile the model
        self.CNNModel.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

        # Train the model
        history = self.CNNModel.fit(X_train, y_train,
                                    epochs=20,
                                    batch_size=128,
                                    validation_split=0.1,
                                    verbose=2)
        
        test_loss, test_acc = self.CNNModel.evaluate(X_test, y_test, verbose=0)
        print(f'\nTest accuracy: {test_acc:.4f}')

    def predictNextPrices(self, priceArray, ticker: str) -> np.ndarray:
        raise NotImplementedError("XGBoost is not implemented yet.")
    
        ## priceArray is array_like
        #fit_results = CurveAnalysis.thirdDegreeFit(priceArray, ticker)
        #features = [m,
        #            fit_results['Coefficients'][0],
        #            fit_results['Coefficients'][1],
        #            fit_results['Coefficients'][2],
        #            fit_results['Coefficients'][3],
        #            fit_results['R_Squared'],
        #            fit_results['Variance']]
        #features = np.array(features).reshape(1, -1)
        #predicted_prices = self.model.predict(features)
        #return predicted_prices[0]
