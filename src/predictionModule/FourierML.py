import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List
import xgboost as xgb
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
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
    __fouriercutoff = 1000
    __spareDatesRatio = 0.05
    __multFactor = 128
    __lenClassInterval = 1
    __daysAfterPrediction = +1
    __numOfMonths = 33


    #__classificationInterval = [0.01*(2*i+1) for i in range(0,lenClassInterval)]
    #__classificationInterval = (np.exp(np.linspace(start=np.log(1.001),stop=np.log(1.15), num=__lenClassInterval))-1).tolist()
    #__classificationInterval = ((np.power(2,range(0,__lenClassInterval))-1)*0.001).tolist()
    chebyNodes = np.cos(( 2*np.arange(1, __lenClassInterval + 2) - 1) * np.pi / (4 * (__lenClassInterval+1)))
    #__classificationInterval = (0.10*(chebyNodes[1:]-chebyNodes[0])/(chebyNodes[-1]-chebyNodes[0])).tolist()
    __classificationInterval = [0.0045]

    ## NOTE! 
    # > len(__classificationInterval) must be equal to __lenClassInterval
    # > __classificationInterval is positive and sorted. 0 is not in __classificationInterval

    def __init__(self, assets: Dict[str, AssetDataPolars], 
                 trainStartDate: pd.Timestamp = None, 
                 trainEndDate: pd.Timestamp = None,
                 testStartDate: pd.Timestamp = None, 
                 testEndDate: pd.Timestamp = None,
                 valStartDate: pd.Timestamp = None, 
                 valEndDate: pd.Timestamp = None,):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets

        self.trainStartDate: pd.Timestamp = trainStartDate
        self.trainEndDate: pd.Timestamp = trainEndDate
        self.testStartDate: pd.Timestamp = testStartDate
        self.testEndDate: pd.Timestamp = testEndDate
        self.valStartDate: pd.Timestamp = valStartDate
        self.valEndDate: pd.Timestamp = valEndDate

        self.metadata['FourierML_params'] = {
            "idxLengthOneMonth" : self.__idxLengthOneMonth,
            "fouriercutoff" : self.__fouriercutoff,
            "spareDatesRatio" : self.__spareDatesRatio,
            "multFactor" : self.__multFactor,
            "lenClassInterval" : self.__lenClassInterval,
            "daysAfterPrediction" : self.__daysAfterPrediction,
            "numOfMonths" : self.__numOfMonths,
            "classificationInterval" : self.__classificationInterval,
        }

    def establishAssetIdx(self) -> Dict:
        # FOR FASTER RUN: Establish index in dataframe to start date
        assetdateIdx = {}
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice["Date"].item(-1) < self.trainStartDate:
                raise ValueError(f"Asset {ticker} history not old enough or startDate ({self.trainStartDate}) too far back. We stop.")
            assetdateIdx[ticker] = DPl(asset.adjClosePrice).getIndex(self.trainStartDate, pd.Timedelta(days=0.7))

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
    def getTargetFromPrice(futureReturn: np.array, percInt: list[float]) -> list[float]:
        """
        Let A be a entry in futurePrices.
        If A is between 0 and the first entry of percInt the method returns on first result entry 0. 
        If A is between the first entry of percInt and the second entry of percInt it returns on second result entry 1, and so on.
        If A is negative it does the above for -A and returns the negativ result entry.
        """
        res = np.zeros(len(futureReturn))
        for j, A in enumerate(futureReturn):
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
    
    def getFeaturesAndTarget(self, asset: AssetDataPolars, pricesArray: pl.Series, date: pd.Timestamp):
        aidx = DPl(asset.adjClosePrice).getNextLowerIndex(date)+1

        m = self.__numOfMonths
        if (aidx - m * self.__idxLengthOneMonth)<0:
            print("Warning! Asset History does not span far enough.")
        if (aidx + self.__daysAfterPrediction)>len(pricesArray):
            print("Warning! Future price does not exist in asset.")

        pastPrices = pricesArray.slice(aidx-m * self.__idxLengthOneMonth, m * self.__idxLengthOneMonth +1).to_numpy()
        futurePrices = pricesArray.slice((aidx+self.__daysAfterPrediction),1).to_numpy()
        
        features = self.getFeaturesFromPrice(pastPrices/pastPrices[-1], 
                                             multFactor=self.__multFactor, 
                                             fouriercutoff=self.__fouriercutoff,
                                             includeLastPrice = True)
        target = self.getTargetFromPrice(futurePrices/pastPrices[-1]-1, self.__classificationInterval)
        target = target[0]

        return features, target, pastPrices/pastPrices[-1], futurePrices/pastPrices[-1]

    def prepareData(self):
        Xtrain = []
        ytrain = []
        Xtest = []
        ytest = []
        Xval = []
        yval = []
        XtrainPrice = []
        ytrainPrice = []
        XtestPrice = []
        ytestPrice = []
        XvalPrice = []
        yvalPrice = []
        lenClassInterval = self.__lenClassInterval
        processedCounter=0

        if self.trainStartDate == None \
             or self.trainEndDate == None \
             or self.testStartDate == None:
            raise ValueError("Data collection time is not defined.")

        #Main Loop
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice is None or not 'AdjClose' in asset.adjClosePrice.columns:
                continue

            print(f"Processing asset: {asset.ticker}.  Processed {processedCounter} out of {len(self.__assets)}.")
            processedCounter += 1

            # Prepare Dates
            #TRAIN
            datesTrain = pd.date_range(self.trainStartDate, self.trainEndDate, freq='B') # 'B' for business days
            spare_datesTrain = pd.DatetimeIndex(np.random.choice(datesTrain, size=int(len(datesTrain)*self.__spareDatesRatio), replace=False)) #unsorted. DO NOT SORT
            #TEST
            if self.testEndDate is None:
                datesTest = pd.date_range(self.testStartDate, self.testStartDate + pd.Timedelta(days=5), freq='B')
                spare_datesTest = [datesTest[0]]
            else:
                datesTest = pd.date_range(self.testStartDate, self.testEndDate, freq='B') # 'B' for business days
                spare_datesTest = pd.DatetimeIndex(np.random.choice(datesTest, size=int(len(datesTest)*self.__spareDatesRatio), replace=False))
            # VALIDATION
            if self.valStartDate is None:
                spare_datesVal = []
            elif self.valEndDate is None:
                datesVal = pd.date_range(self.valStartDate, self.valStartDate + pd.Timedelta(days=5), freq='B')
                spare_datesVal = [datesVal[0]]
            else:
                datesVal = pd.date_range(self.valStartDate, self.valEndDate, freq='B') # 'B' for business days
                spare_datesVal = pd.DatetimeIndex(np.random.choice(datesVal, size=int(len(datesVal)*self.__spareDatesRatio), replace=False))
        

            pricesArray = asset.adjClosePrice['AdjClose']

            # Prepare Train Data
            for date in spare_datesTrain:
                features, target, featuresTimeSeries, targetTimeSeries = self.getFeaturesAndTarget(asset, pricesArray, date)

                Xtrain.append(features)
                ytrain.append(target)
                XtrainPrice.append(featuresTimeSeries)
                ytrainPrice.append(targetTimeSeries)

            #Prepare Test Data
            for date in spare_datesTest:
                features, target, featuresTimeSeries, targetTimeSeries = self.getFeaturesAndTarget(asset, pricesArray, date)

                Xtest.append(features)
                ytest.append(target)
                XtestPrice.append(featuresTimeSeries)
                ytestPrice.append(targetTimeSeries)

            #Prepare Val Data
            for date in spare_datesVal:
                features, target, featuresTimeSeries, targetTimeSeries = self.getFeaturesAndTarget(asset, pricesArray, date)

                Xval.append(features)
                yval.append(target)
                XvalPrice.append(featuresTimeSeries)
                yvalPrice.append(targetTimeSeries)

        self.dataIsPrepared = True
        self.X_train = np.array(Xtrain)
        self.y_train = np.array(ytrain).astype(int)+lenClassInterval
        self.X_test = np.array(Xtest)
        self.y_test = np.array(ytest).astype(int)+lenClassInterval
        self.X_val = np.array(Xval)
        self.y_val = np.array(yval).astype(int)+lenClassInterval
        self.X_train_timeseries = np.array(XtrainPrice)
        self.y_train_timeseries = np.array(ytrainPrice)
        self.X_test_timeseries = np.array(XtestPrice)
        self.y_test_timeseries = np.array(ytestPrice)

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test)
        self.X_train_timeseries, self.y_train_timeseries = shuffle(self.X_train_timeseries, self.y_train_timeseries)
        self.X_test_timeseries, self.y_test_timeseries = shuffle(self.X_test_timeseries, self.y_test_timeseries)

    def traintestXGBModel(self, xgb_params=None):
        if not self.dataIsPrepared:
            self.prepareData()

        # Split the data
        X_val = self.X_test
        y_val = self.y_test
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        # Define XGBoost parameters if not provided
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.05
            }
        self.metadata['XGBoostModel_params'] = xgb_params

        # Initialize and train XGBoost model
        self.XGBoostModel = xgb.XGBClassifier(**xgb_params)
        self.XGBoostModel.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )

        # Make predictions
        y_pred = self.XGBoostModel.predict(X_val)
        y_pred_proba = self.XGBoostModel.predict_proba(X_val)

        # Calculate accuracy
        test_acc = accuracy_score(y_val, y_pred)

        # Calculate log loss
        test_loss = log_loss(y_val, y_pred_proba)

        self.metadata['XGBoostModel_Test_accuracy'] = test_acc
        self.metadata['XGBoostModel_Test_log_loss'] = test_loss
        print(f'\nTest accuracy: {test_acc:.4f}')
        print(f'Test log loss: {test_loss:.4f}')

    def traintestCNNModel(self):
        if not self.dataIsPrepared:
            self.prepareData()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=.2)

        input_shape = X_train.shape[1]
        output_shape = 2*self.__lenClassInterval+1
        y_cat_train = to_categorical(y_train)
        y_cat_test = to_categorical(y_test)

        # Define CNN parameters
        cnn_params = {
            'layers': [
                {'units': 128, 'activation': 'relu'},
                {'dropout': 0.2},
                {'units': 64, 'activation': 'relu'},
                {'units': 2 * self.__lenClassInterval + 1, 'activation': 'softmax'}
            ],
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            'epochs': 20,
            'batch_size': 128
        }
        self.metadata['CNNModel_params'] = cnn_params

        # Define the model
        self.CNNModel = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_shape, activation='sigmoid')
        ])

        # Compile the model
        self.CNNModel.compile(optimizer=cnn_params['optimizer'],
                              loss=cnn_params['loss'],
                              metrics=cnn_params['metrics'])

        # Train the model
        history = self.CNNModel.fit(
            self.X_train, to_categorical(self.y_train, num_classes=2 * self.__lenClassInterval + 1),
            epochs=cnn_params['epochs'],
            batch_size=cnn_params['batch_size'],
            validation_split=0.2,
            verbose=2
        )
        
        test_loss, test_acc = self.CNNModel.evaluate(X_test, y_cat_test, verbose=0)
        print(f'\nTest accuracy: {test_acc:.4f}')

    def traintestLGBMModel(self):
        if not self.dataIsPrepared:
            self.prepareData()
        
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=.2)

        y_cat_train = to_categorical(y_train)
        y_cat_test = to_categorical(y_test)

        # Define LGBM parameters
        lgbm_params = {
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'max_depth': 5,
            'num_leaves': 32,
            'colsample_bytree': 0.1,
            'early_stopping_rounds': 100
        }
        self.metadata['LGBMModel_params'] = lgbm_params

        # Initialize and train LGBM model
        self.LGBMModel = lgb.LGBMClassifier(**lgbm_params)
        # Define the model
        self.LGBMModel = lgb.LGBMRegressor(
            n_estimator=lgbm_params["n_estimators"],
            learning_rate = lgbm_params["learning_rate"],
            max_depth=lgbm_params["max_depth"],
            num_leaves=lgbm_params["num_leaves"],
            colsample_bytree=lgbm_params["colsample_bytree"]
        )

        # Train the model
        self.LGBMModel.fit(X_train, y_cat_train,
                        eval_set=[(self.X_test, self.y_test)],
                        early_stopping_rounds=lgbm_params["early_stopping_rounds"],
                        verbose=100)
        
        test_loss, test_acc = self.CNNModel.evaluate(X_test, y_cat_test, verbose=100)
        print(f'\nTest accuracy: {test_acc:.4f}')

    def traintestRPModel(self, rp_params=None):
        if not self.dataIsPrepared:
            self.prepareData()
        
        # Split the data
        X_val = self.X_test
        y_val = self.y_test
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=.1)

        # Define XGBoost parameters if not provided
        if rp_params is None:
            # Define RPModel parameters
            rp_params = {
                'num_random_features': 2000,
                'regularization': 30,
                'max_iter': 10,
                'verbose': True,
                'random_state': None
            }
        self.metadata['RPModel_params'] = rp_params

        # Initialize and train RPModel
        self.RPModel = rpc(**rp_params)

        # Train the model
        self.RPModel.fit(X_train, y_train)

        self.RPModel.tune_regularization(X_test, y_test, low_start=1, high_start=10, max_iter=10)
        print(self.RPModel.g)
        test_acc = self.RPModel.compute_accuracy(g=self.RPModel.g, Y_test = y_val, X_test=X_val)
        self.metadata['RPModel_Test_accuracy'] = test_acc
        print(f'\nTest accuracy: {test_acc:.4f}')
    
    def traintestLSTMModel(self):
        if not self.dataIsPrepared:
            self.prepareData()

        # Define LSTM parameters
        lstm_params = {
            'units': 128,
            'dropout': 0.2,
            'dense_units': 64,
            'output_units': 2 * self.__lenClassInterval + 1,
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            'epochs': 20,
            'batch_size': 128
        }
        self.metadata['LSTMModel_params'] = lstm_params

        # Initialize and compile LSTM model
        self.LSTMModel = models.Sequential([
            layers.LSTM(lstm_params['units'], input_shape=(1, self.X_train.shape[1])),
            layers.Dropout(lstm_params['dropout']),
            layers.Dense(lstm_params['dense_units'], activation=lstm_params['activation']),
            layers.Dense(lstm_params['output_units'], activation='softmax')
        ])
        self.LSTMModel.compile(optimizer=lstm_params['optimizer'],
                               loss=lstm_params['loss'],
                               metrics=lstm_params['metrics'])

        # Train the model
        history = self.LSTMModel.fit(
            self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1])),
            to_categorical(self.y_train, num_classes=lstm_params['output_units']),
            epochs=lstm_params['epochs'],
            batch_size=lstm_params['batch_size'],
            validation_split=0.2,
            verbose=2
        )

        # Evaluate the model
        test_loss, test_acc = self.LSTMModel.evaluate(X_test, y_cat_test, verbose=0)
        print(f'\nTest accuracy: {test_acc:.4f}')


    def predictNextPrices(self, priceArray: np.array, ticker: str) -> int:
        # priceArray is array_like and for at least two years
        if len(priceArray) < self.__numOfMonths * self.__idxLengthOneMonth -1:
            print("priceArray might be too short.")
        if self.CNNModel is None:
            raise ValueError('No model has been trained to save.')
        
        features = self.getFeaturesFromPrice(priceArray/priceArray[-1], multFactor=self.__multFactor, fouriercutoff=self.__fouriercutoff)
        predicted_box = self.CNNModel.predict(features)-self.__lenClassInterval
        return int(predicted_box)