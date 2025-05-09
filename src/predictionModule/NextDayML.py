import numpy as np
import pandas as pd
import polars as pl
import bisect
import holidays
from typing import Dict, List
from sklearn.utils import shuffle
import pycountry
from sklearn.preprocessing import MinMaxScaler


from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.mathTools.TAIndicators import TAIndicators
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.predictionModule.IML import IML
from src.featureAlchemy.FeatureMain import FeatureMain

class NextDayML(IML):
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 5,
        'multFactor': 6,
        'daysAfterPrediction': 1,
        'monthsHorizon': 13,
        'timesteps': 5,
        'classificationInterval': [-0.0045, 0.0045], 
        'averageOverDays': 5,
    }

    def __init__(self, assets: Dict[str, AssetDataPolars], 
                 trainDates: pd.DatetimeIndex = None,
                 valDates: pd.DatetimeIndex = None,
                 testDates: pd.DatetimeIndex = None,
                 params: dict = None,
                 enableTimeSeries = True):
        super().__init__()
        self.__assets: Dict[str, AssetDataPolars] = assets

        self.trainDates: pd.DatetimeIndex = trainDates
        self.valDates: pd.DatetimeIndex = valDates
        self.testDates: pd.DatetimeIndex = testDates
        
        self.enableTimeSeries = enableTimeSeries
        self.lagList = [1,2,3,5,10,21,63, 121, 210, 21*12]
        self.featureColumnNames = []
        
        # Update default parameters with any provided parameters
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

        # Assign parameters to instance variables
        self.idxLengthOneMonth = self.params['idxLengthOneMonth']
        self.fouriercutoff = self.params['fouriercutoff']
        self.multFactor = self.params['multFactor']
        self.daysAfterPrediction = self.params['daysAfterPrediction']
        self.monthsHorizon = self.params['monthsHorizon']
        self.classificationInterval = self.params['classificationInterval']
        self.timesteps = self.params['timesteps'] 
        self.averageOverDays: int = self.params['averageOverDays']
        
        if self.daysAfterPrediction < self.averageOverDays//2:
            raise ValueError("daysAfterPrediction should be greater than averageOverDays//2.")
        
        # Store parameters in metadata
        self.metadata['NextDayML_params'] = self.params
    
    @staticmethod
    def getTargetFromPrice(futureReturn: np.array, sorted_array: list[float]) -> list[float]:
        """
        Args:
            futureReturn (np.array): array of returns. (Ought to be around 0)
            sorted_array (list[float]): sorted list of floats. (Ought to be centered at 0)

        Returns:
            list[float]: for every entry in futureReturn, checks what the index of the next upper value in percInt is.
        """
        
        indices = [bisect.bisect_right(sorted_array, value) for value in futureReturn]

        return indices
    
    def getFeaturesAndTarget(self, asset: AssetDataPolars, featureMain: FeatureMain, date: pd.Timestamp, aidx: int):
        m = self.monthsHorizon
        numTimesteps = self.timesteps
        if (aidx - m * self.idxLengthOneMonth-1-numTimesteps)<0:
            print("Warning! Asset History does not span far enough.")
        
        curPrice = asset.adjClosePrice["AdjClose"].item(aidx)
        futurePrices = (
            asset.adjClosePrice["AdjClose"]
                .slice(aidx + self.daysAfterPrediction - self.averageOverDays//2, self.averageOverDays).to_numpy()
        )
        futureMeanPrice = futurePrices.mean()
        futureMeanPriceScaled = futureMeanPrice/curPrice
        
        features = featureMain.apply(date, idx=aidx)
        features_timeseries = featureMain.apply(date, idx=aidx)
        
        target = self.getTargetFromPrice([futureMeanPriceScaled-1], self.classificationInterval)
        target = target[0]

        return features, target, features_timeseries, [futureMeanPrice]

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
        processedCounter=0

        if self.trainDates is None \
             or self.valDates is None \
             or self.testDates is None:
            raise ValueError("Data collection time is not defined.")

        if not self.trainDates.intersection(self.valDates).empty:
            raise ValueError("There are overlapping dates between Train-Validation Dates and Val Dates.")
        if not (self.trainDates.union(self.valDates)).intersection(self.testDates).empty:
                raise ValueError("There are overlapping dates between Train-Validation Dates and Test Dates.")

        #Main Loop
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice is None or not 'AdjClose' in asset.adjClosePrice.columns:
                continue

            print(f"Processing asset: {asset.ticker}.  Processed {processedCounter} out of {len(self.__assets)}.")
            processedCounter += 1
            
            params = {
                'idxLengthOneMonth': self.idxLengthOneMonth,
                'fouriercutoff': self.fouriercutoff,
                'multFactor': self.multFactor,
                'monthsHorizon': self.monthsHorizon,
                'timesteps': self.timesteps,
            }
            featureMain = FeatureMain(
                asset, 
                min([self.trainDates.min(),self.valDates.min(),self.testDates.min()]), 
                max([self.trainDates.max(),self.valDates.max(),self.testDates.max()]),
                lagList = self.lagList, 
                params=params,
                enableTimeSeries = self.enableTimeSeries
            )
            
            if self.featureColumnNames == []:
                self.featureColumnNames = featureMain.getFeatureNames()
            elif self.featureColumnNames != featureMain.getFeatureNames():
                raise ValueError("Feature column names are not consistent across assets.")

            # Prepare Train Data And Val Dates
            trainValDates = self.trainDates.union(self.valDates)
            for date in trainValDates:
                aidx = DPl(asset.adjClosePrice).getNextLowerOrEqualIndex(date)
                if asset.shareprice["Date"].item(aidx) != date:
                    continue
                features, target, featuresTimeSeries, targetTimeSeries = self.getFeaturesAndTarget(asset, featureMain, date, aidx)

                if date in self.trainDates:
                    Xtrain.append(features)
                    ytrain.append(target)
                    XtrainPrice.append(featuresTimeSeries)
                    ytrainPrice.append(targetTimeSeries)
                if date in self.valDates:
                    Xval.append(features)
                    yval.append(target)
                    XvalPrice.append(featuresTimeSeries)
                    yvalPrice.append(targetTimeSeries)

            #Prepare Test Data
            for date in self.testDates:
                aidx = DPl(asset.adjClosePrice).getNextLowerOrEqualIndex(date)
                if asset.shareprice["Date"].item(aidx) != date:
                    continue
                features, target, featuresTimeSeries, targetTimeSeries = self.getFeaturesAndTarget(asset, featureMain, date, aidx)

                Xtest.append(features)
                ytest.append(target)
                XtestPrice.append(featuresTimeSeries)
                ytestPrice.append(targetTimeSeries)

        self.X_train = np.array(Xtrain)
        self.y_train = np.array(ytrain).astype(int)
        self.X_test = np.array(Xtest)
        self.y_test = np.array(ytest).astype(int)
        self.X_val = np.array(Xval)
        self.y_val = np.array(yval).astype(int)
        self.X_train_timeseries = np.array(XtrainPrice)
        self.y_train_timeseries = np.array(ytrainPrice)
        self.X_test_timeseries = np.array(XtestPrice)
        self.y_test_timeseries = np.array(ytestPrice)
        self.X_val_timeseries = np.array(XvalPrice)
        self.y_val_timeseries = np.array(yvalPrice)

        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test)
        self.X_val, self.y_val = shuffle(self.X_val, self.y_val)
        
        self.X_val_timeseries, self.y_val_timeseries = shuffle(self.X_val_timeseries, self.y_val_timeseries)
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
            features, _, _, _ = self.getFeaturesAndTarget(
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