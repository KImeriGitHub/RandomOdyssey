import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List
import xgboost as xgb
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import lightgbm as lgb


from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.mathTools.RandomProjectionClassifier import RandomProjectionClassifier as rpc
from src.predictionModule.IML import IML

class FourierML(IML):
    __idxLengthOneMonth = 21
    __fouriercutoff = 20
    __spareDatesRatio = 0.3
    __multFactor = 32
    __lenClassInterval = 1
    __daysAfterPrediction = 1
    __numOfMonths = 6


    #__classificationInterval = [0.01*(2*i+1) for i in range(0,lenClassInterval)]
    #__classificationInterval = (np.exp(np.linspace(start=np.log(1.001),stop=np.log(1.15), num=__lenClassInterval))-1).tolist()
    #__classificationInterval = ((np.power(2,range(0,__lenClassInterval))-1)*0.001).tolist()
    chebyNodes = np.cos(( 2*np.arange(1, __lenClassInterval + 2) - 1) * np.pi / (4 * (__lenClassInterval+1)))
    #__classificationInterval = (0.10*(chebyNodes[1:]-chebyNodes[0])/(chebyNodes[-1]-chebyNodes[0])).tolist()
    __classificationInterval = [0.0045]

    ## NOTE! 
    # > len(__classificationInterval) must be equal to __lenClassInterval
    # > __classificationInterval is positive and sorted. 0 is not in __classificationInterval

    def __init__(self, assets: Dict[str, AssetDataPolars], trainStartDate: pd.Timestamp, trainEndDate: pd.Timestamp):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets
        self.__trainStartDate: pd.Timestamp = trainStartDate
        self.__trainEndDate: pd.Timestamp = trainEndDate

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
    def getFeaturesFromPrice(pastPrices: np.array, includeLastPrice = False, multFactor: int = 8, fouriercutoff: int = 100) -> list[float]:
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
        fourierInput = np.concatenate((skewedPrices,np.flipud(-skewedPrices[:(n-1)])))
        cs = CubicSpline(np.arange(len(fourierInput)), fourierInput, bc_type='periodic')
        fourierInputSpline = cs(np.linspace(0, len(fourierInput)-1, 1 + (len(fourierInput) - 1) * multFactor))
        fourierInputSmooth = gaussian_filter1d(fourierInputSpline, sigma=np.max([multFactor//4,1]), mode = "wrap")
        res_cos, res_sin = SeriesExpansion.getFourierConst(fourierInputSmooth)
        res_cos=res_cos.T.real.flatten().tolist()
        res_sin=res_sin.T.real.flatten().tolist()

        if len(res_cos) < fouriercutoff:
            raise Warning("fouriercutoff is bigger than the array itself.")

        features = []
        if includeLastPrice:
            features.append(fxend)
            features.append((fxend-fx0)/(n-1))
        else:
            features.append((1-fx0/fxend)/(n-1))

        endIdx = np.min([len(res_cos), fouriercutoff//2])
        features.extend(res_cos[1:endIdx])
        features.extend(res_sin[1:endIdx])

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
        lenClassInterval = self.__lenClassInterval
        classificationIntervals = self.__classificationInterval
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
            spare_dates = pd.DatetimeIndex(np.random.choice(dates, size=int(len(dates)*self.__spareDatesRatio), replace=False)) #unsorted. DO NOT SORT
            for date in spare_dates:
                if not abs(datesArray.item(assetdateIdx[ticker]) - date) <= pd.Timedelta(hours=18):
                    assetdateIdx[ticker] = DPl(asset.adjClosePrice).getNextLowerIndex(date)+1
                
                m = self.__numOfMonths
                aidx = assetdateIdx[ticker]
                if (aidx - m * self.__idxLengthOneMonth)<0:
                    continue
                pastPrices = pricesArray.slice(aidx-m * self.__idxLengthOneMonth, m * self.__idxLengthOneMonth +1).to_numpy()
                futurePrices = pricesArray.slice((aidx+self.__daysAfterPrediction),1).to_numpy()
                
                features = self.getFeaturesFromPrice(pastPrices, multFactor=self.__multFactor, 
                                                     fouriercutoff=self.__fouriercutoff,
                                                     includeLastPrice = True)
                target = self.__getTargetFromPrice(futurePrices/pastPrices[-1]-1, classificationIntervals)
                target = target[0]
                
                if not np.isfinite(pastPrices).all() \
                    or not np.isfinite(futurePrices).all()\
                    or not np.isfinite(features).all():
                    raise ValueError("Stock Prices are not complete")
                X.append(features)
                y.append(target)

        self.__dataIsPrepared = True
        self.X = np.array(X)
        self.y = np.array(y).astype(int)+lenClassInterval

    def traintestCNNModel(self):
        if not self.__dataIsPrepared:
            self.prepareData()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2)

        input_shape = X_train.shape[1]
        output_shape = 2*self.__lenClassInterval+1
        y_cat_train = to_categorical(y_train)
        y_cat_test = to_categorical(y_test)

        # Define the model
        self.CNNModel = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_shape, activation='sigmoid')
        ])

        # Compile the model
        self.CNNModel.compile(optimizer='adam',
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

        # Train the model
        history = self.CNNModel.fit(X_train, y_cat_train,
                                    epochs=20,
                                    batch_size=128,
                                    validation_split=0.1,
                                    verbose=2)
        
        test_loss, test_acc = self.CNNModel.evaluate(X_test, y_cat_test, verbose=0)
        print(f'\nTest accuracy: {test_acc:.4f}')

    def traintestCNNModel(self):
        if not self.__dataIsPrepared:
            self.prepareData()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2)

        y_cat_train = to_categorical(y_train)
        y_cat_test = to_categorical(y_test)

        # Define the model
        self.LGBMModel = lgb.LGBMRegressor(
            n_estimator=2000,
            learning_rate = 0.01,
            max_depth=5,
            num_leaves=2 ** 5,
            colsample_bytree=0.1
        )

        # Train the model
        self.LGBMModel.fit(X_train, y_cat_train)
        
        test_loss, test_acc = self.CNNModel.evaluate(X_test, y_cat_test, verbose=0)
        print(f'\nTest accuracy: {test_acc:.4f}')

    def traintestRPModel(self):
        if not self.__dataIsPrepared:
            self.prepareData()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=.2)

        # Initialize the classifier
        self.RPCModel = rpc(
            num_random_features=2000,
            regularization=30,
            max_iter=10,
            verbose=True,
            random_state=None
        )

        # Train the model
        self.RPCModel.fit(X_train, y_train)


    def predictNextPrices(self, priceArray: np.array, ticker: str) -> int:
        # priceArray is array_like and for at least two years
        if len(priceArray) < self.__numOfMonths * self.__idxLengthOneMonth -1:
            print("priceArray might be too short.")
        if self.CNNModel is None:
            raise ValueError('No model has been trained to save.')
        
        features = self.getFeaturesFromPrice(priceArray/priceArray[-1], multFactor=self.__multFactor, fouriercutoff=self.__fouriercutoff)
        predicted_box = self.CNNModel.predict(features)-self.__lenClassInterval
        return int(predicted_box)
