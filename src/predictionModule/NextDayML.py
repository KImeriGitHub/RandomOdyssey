import numpy as np
import pandas as pd
import polars as pl
import bisect
from typing import Dict, List
import xgboost as xgb
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from ta import add_all_ta_features

from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.mathTools.RandomProjectionClassifier import RandomProjectionClassifier as rpc
from src.predictionModule.IML import IML

class NextDayML(IML):
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 25,
        'spareDatesRatio': 0.5,
        'multFactor': 8,
        'daysAfterPrediction': 1,
        'monthsHorizon': 13,
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
        if params is None:
            self.params = self.DEFAULT_PARAMS
        else:
            self.params = params

        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.spareDatesRatio = self.params['spareDatesRatio']
        self.multFactor = self.params['multFactor']
        self.daysAfterPrediction = self.params['daysAfterPrediction']
        self.monthsHorizon = self.params['monthsHorizon']
        self.classificationInterval = self.params['classificationInterval'] 

        # Store parameters in metadata
        self.metadata['NextDayML_params'] = self.params

    @staticmethod
    def getFourierFeaturesFromPrice(pastPrices: np.array, multFactor: int = 8, fouriercutoff: int = 25) -> list[float]:
        """
        Args:
            pastPrices (np.array): slice of prices
            multFactor (int, optional): adds that many points in between. Defaults to 8.
            fouriercutoff (int, optional): from all fourier coeffs cutoff that many. Defaults to 25.

        Raises:
            ValueError: if pastPrices invalid
            Warning: if fouriercutoff too high

        Returns:
            list[float]: features: first one is average increase, then cos coeffs, then sin coeffs
        """
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
        features.append((1-fx0/fxend)/(n-1))
        endIdx = np.min([len(res_cos), fouriercutoff])
        features.extend(res_cos[1:endIdx])
        features.extend(res_sin[1:endIdx])
        
        # Add fourier approximation error to the features
        N = len(fourierInputSmooth)
        t = np.linspace(-np.pi, np.pi, N, endpoint=False)
        f_reconstructed = np.zeros(N)
        for n in range(0,endIdx):
            f_reconstructed += res_cos[n] * np.cos(n * t) + res_sin[n] * np.sin(n * t)
        # Calculate the mean squared error between the original and reconstructed functions
        absErrorVector = np.abs(fourierInputSmooth - f_reconstructed) / 2.0
        features.append(np.mean(absErrorVector ** 2)) # squared mean error
        features.append(np.mean(absErrorVector)) # absolute mean error
        
        return features
    
    @staticmethod
    def getTargetFromPrice(futureReturn: np.array, sorted_array: list[float]) -> list[float]:
        """
        Args:
            futureReturn (np.array): array of returns. (Ought to be around 1)
            sorted_array (list[float]): sorted list of floats. (Ought to be centered at 0)

        Returns:
            list[float]: for every entry in futureReturn, checks what the index of the next upper value in percInt is.
        """
        
        indices = [bisect.bisect_right(sorted_array, value) for value in futureReturn]

        return indices
    
    def getTAFeatures(self, row:list):
        features = row[1:]
        return features
    
    def getFeaturesAndTarget(self, asset: AssetDataPolars, pricesArray: pl.Series, asset_taExt: pd.DataFrame, date: pd.Timestamp):
        aidx = DPl(asset.adjClosePrice).getNextLowerIndex(date)+1

        m = self.monthsHorizon
        if (aidx - m * self.idxLengthOneMonth)<0:
            print("Warning! Asset History does not span far enough.")
        if (aidx + self.daysAfterPrediction)>len(pricesArray):
            print("Warning! Future price does not exist in asset.")

        pastPrices = pricesArray.slice(aidx-m * self.idxLengthOneMonth, m * self.idxLengthOneMonth +1).to_numpy()
        pastPricesScaled = pastPrices/pastPrices[-1]
        futurePrices = pricesArray.slice((aidx+self.daysAfterPrediction),1).to_numpy()
        futurePricesScaled = futurePrices/pastPrices[-1]
        
        taRow = asset_taExt.iloc[aidx, :].values.tolist()
        
        features = []
        
        #Fourier Features
        fourierFeatures = self.getFourierFeaturesFromPrice(pastPricesScaled, 
                                             multFactor=self.multFactor, 
                                             fouriercutoff=self.fouriercutoff,
                                             includeLastPrice = False)
        features.extend(fourierFeatures)
        
        #TA Features
        taFeatures = self.getTAFeatures(taRow)
        features.extend(taFeatures)
        
        
        target = self.getTargetFromPrice(futurePricesScaled-1, self.classificationInterval)
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
            
            # Technical Analysis extension
            asset_taExt = add_all_ta_features(asset.shareprice.to_pandas(), open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

            # Prepare Dates
            #TRAIN
            datesTrain = pd.date_range(self.trainStartDate, self.trainEndDate, freq='B') # 'B' for business days
            spare_datesTrain = pd.DatetimeIndex(np.random.choice(datesTrain, size=int(len(datesTrain)*self.spareDatesRatio), replace=False)) 
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
                features, target, featuresTimeSeries, targetTimeSeries = self.getFeaturesAndTarget(asset, pricesArray, asset_taExt, date)

                Xtrain.append(features)
                ytrain.append(target)
                XtrainPrice.append(featuresTimeSeries)
                ytrainPrice.append(targetTimeSeries)

            #Prepare Test Data
            for date in spare_datesTest:
                features, target, featuresTimeSeries, targetTimeSeries = self.getFeaturesAndTarget(asset, pricesArray, asset_taExt, date)

                Xtest.append(features)
                ytest.append(target)
                XtestPrice.append(featuresTimeSeries)
                ytestPrice.append(targetTimeSeries)

            #Prepare Val Data
            for date in spare_datesVal:
                features, target, featuresTimeSeries, targetTimeSeries = self.getFeaturesAndTarget(asset, pricesArray, asset_taExt, date)

                Xval.append(features)
                yval.append(target)
                XvalPrice.append(featuresTimeSeries)
                yvalPrice.append(targetTimeSeries)

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
        
        self.dataIsPrepared = True


    def predictNextPrices(self, priceArray: np.ndarray, isRegression: bool = False):
        if len(priceArray) < self.monthsHorizon * self.idxLengthOneMonth -1:
            print("priceArray might be too short.")
        if isRegression:
            # Ensure LSTM model is trained
            if 'LSTMModel_loss' not in self.metadata:
                raise ValueError("LSTM model has not been trained.")
            
            if self.LSTMModel is None:
                raise ValueError("LSTM model instance is not available.")
            
            # Prepare timeseries features
            past_prices = priceArray[-(self.monthsHorizon * self.idxLengthOneMonth):]
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