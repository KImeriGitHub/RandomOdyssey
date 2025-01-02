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


from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.mathTools.RandomProjectionClassifier import RandomProjectionClassifier as rpc
from src.predictionModule.IML import IML

#############
# DEPRECATED
#############
class FourierML(IML):
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 4000,
        'spareDatesRatio': 0.5,
        'multFactor': 256,
        'daysAfterPrediction': +1,
        'numOfMonths': 13,
        'classificationInterval': [0.0045], 
    }

    def __init__(self, assets: Dict[str, AssetDataPolars], 
                 trainStartDate: pd.Timestamp = None, 
                 trainEndDate: pd.Timestamp = None,
                 testStartDate: pd.Timestamp = None, 
                 testEndDate: pd.Timestamp = None,
                 valStartDate: pd.Timestamp = None, 
                 valEndDate: pd.Timestamp = None,
                 params: dict = None):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets

        self.trainStartDate: pd.Timestamp = trainStartDate
        self.trainEndDate: pd.Timestamp = trainEndDate
        self.testStartDate: pd.Timestamp = testStartDate
        self.testEndDate: pd.Timestamp = testEndDate
        self.valStartDate: pd.Timestamp = valStartDate
        self.valEndDate: pd.Timestamp = valEndDate
        
        # Update default parameters with any provided parameters
        self.params = self.DEFAULT_PARAMS
        if params is not None:
            self.params.update(params)

        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.spareDatesRatio = self.params['spareDatesRatio']
        self.multFactor = self.params['multFactor']
        self.daysAfterPrediction = self.params['daysAfterPrediction']
        self.numOfMonths = self.params['numOfMonths']
        self.classificationInterval = self.params['classificationInterval'] 

        # Store parameters in metadata
        self.metadata['FourierML_params'] = self.params

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
        aidx = DPl(asset.adjClosePrice).getNextLowerOrEqualIndex(date)

        m = self.numOfMonths
        if (aidx - m * self.idxLengthOneMonth)<0:
            print("Warning! Asset History does not span far enough.")
        if (aidx + self.daysAfterPrediction)>len(pricesArray):
            print("Warning! Future price does not exist in asset.")

        pastPrices = pricesArray.slice(aidx-m * self.idxLengthOneMonth, m * self.idxLengthOneMonth +1).to_numpy()
        futurePrices = pricesArray.slice((aidx+self.daysAfterPrediction),1).to_numpy()
        
        features = self.getFeaturesFromPrice(pastPrices/pastPrices[-1], 
                                             multFactor=self.multFactor, 
                                             fouriercutoff=self.fouriercutoff,
                                             includeLastPrice = False)
        target = self.getTargetFromPrice(futurePrices/pastPrices[-1]-1, self.classificationInterval)
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
        lenClassInterval = len(self.classificationInterval)
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
            spare_datesTrain = pd.DatetimeIndex(np.random.choice(datesTrain, size=int(len(datesTrain)*self.spareDatesRatio), replace=False)) #unsorted. DO NOT SORT
            #TEST
            if self.testEndDate is None:
                datesTest = pd.date_range(self.testStartDate, self.testStartDate + pd.Timedelta(days=5), freq='B')
                spare_datesTest = [datesTest[0]]
            else:
                datesTest = pd.date_range(self.testStartDate, self.testEndDate, freq='B') # 'B' for business days
                spare_datesTest = pd.DatetimeIndex(np.random.choice(datesTest, size=np.max([int(len(datesTest)*self.spareDatesRatio),1]), replace=False))
            # VALIDATION
            if self.valStartDate is None:
                spare_datesVal = []
            elif self.valEndDate is None:
                datesVal = pd.date_range(self.valStartDate, self.valStartDate + pd.Timedelta(days=5), freq='B')
                spare_datesVal = [datesVal[0]]
            else:
                datesVal = pd.date_range(self.valStartDate, self.valEndDate, freq='B') # 'B' for business days
                spare_datesVal = pd.DatetimeIndex(np.random.choice(datesVal, size=np.max([int(len(datesVal)*self.spareDatesRatio),1]), replace=False))
        
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


    def predictNextPrices(self, priceArray: np.ndarray, isRegression: bool = False):
        if len(priceArray) < self.numOfMonths * self.idxLengthOneMonth -1:
            print("priceArray might be too short.")
        if isRegression:
            # Ensure LSTM model is trained
            if 'LSTMModel_loss' not in self.metadata:
                raise ValueError("LSTM model has not been trained.")
            
            if self.LSTMModel is None:
                raise ValueError("LSTM model instance is not available.")
            
            # Prepare timeseries features
            past_prices = priceArray[-(self.numOfMonths * self.idxLengthOneMonth):]
            past_prices_normalized = past_prices / past_prices[-1]
            
            # Scale the data
            scaled_X = self.scaler_X.transform([past_prices_normalized])
            
            # Reshape for LSTM: (samples, timesteps, features)
            timesteps = 1
            num_features = scaled_X.shape[1] // timesteps
            scaled_X = scaled_X.reshape((scaled_X.shape[0], timesteps, num_features))
            
            # Predict with LSTM
            predicted = self.LSTMModel.predict(scaled_X)
            
            # Inverse scale the prediction if necessary
            predicted_price_change = self.scaler_y.inverse_transform(predicted)[0][0]
            return predicted_price_change

        else:
            # Prepare classification features
            features = self.getFeaturesFromPrice(
                pastPrices=priceArray / priceArray[-1], 
                multFactor=self.multFactor, 
                fouriercutoff=self.fouriercutoff,
                includeLastPrice=False
            )
            features = np.array(features).reshape(1, -1)
            
            # Identify the best classifier based on accuracy
            classifiers = ['XGBoostModel', 'CNNModel', 'LGBMModel', 'RPModel']
            best_model = None
            best_accuracy = -np.inf
            
            for model_name in classifiers:
                acc_key = f'{model_name}_accuracy'
                if acc_key in self.metadata:
                    acc = self.metadata[acc_key]
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model = getattr(self, model_name)
            
            if best_model is None:
                raise ValueError("No trained classifiers available for prediction.")
            
            # Predict with the best classifier
            predicted_class = best_model.predict(features)[0]
            return int(predicted_class)