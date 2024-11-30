import numpy as np
import pandas as pd
import polars as pl
import bisect
import holidays
from typing import Dict, List
import xgboost as xgb
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from ta import add_all_ta_features
import holidays
import pycountry
from sympy import isprime


from src.common.AssetDataPolars import AssetDataPolars
from src.mathTools.SeriesExpansion import SeriesExpansion
from src.common.DataFrameTimeOperations import DataFrameTimeOperationsPolars as DPl
from src.mathTools.RandomProjectionClassifier import RandomProjectionClassifier as rpc
from src.predictionModule.IML import IML

class NextDayML(IML):
    # Class-level default parameters
    DEFAULT_PARAMS = {
        'idxLengthOneMonth': 21,
        'fouriercutoff': 15,
        'spareDatesRatio': 0.5,
        'multFactor': 8,
        'daysAfterPrediction': 1,
        'monthsHorizon': 13,
        'timesteps': 25,
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
        self.monthsHorizon = self.params['monthsHorizon']
        self.classificationInterval = self.params['classificationInterval']
        self.timesteps = self.params['timesteps'] 

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
        features.append((1-fx0/fxend)/(n-1)) if abs(fxend)>1e-10 else features.append(0)
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
        features.append(np.sqrt(np.mean(absErrorVector))) # root absolute mean error
        
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
        features = np.clip(features, -1e16, 1e16)
        return features
    
    def getCategoryFeatures(self, asset: AssetDataPolars):
        categories = [
            'other', 'industrials', 'healthcare', 'technology', 'utilities', 
            'financial-services', 'basic-materials', 'real-estate', 
            'consumer-defensive', 'energy', 'communication-services', 
            'consumer-cyclical'
        ]

        # Create a mapping dictionary
        category_to_num = {category: idx for idx, category in enumerate(categories)}
        
        return [category_to_num.get(asset.about.get('sectorKey','other'),0)]
    
    def __USHolidays(self):
        country_holidays = holidays.CountryHoliday('US')
        for y in range(self.trainStartDate.year-1, self.valEndDate.year+2):
            country_holidays.get(f"{y}")
        return sorted(country_holidays.keys())
        
    def getSeasonalFeatures(self, timestamp: pd.Timestamp, country: str = 'US') -> dict:
        """
        Extracts comprehensive date-related features for a given pd.Timestamp.
        Parameters:
            timestamp (pd.Timestamp): The date to extract features from.
            country (str): The country code for holiday determination (default: 'US').
        Returns:
            dict: A dictionary containing the extracted date features.
        """
        if not isinstance(timestamp, pd.Timestamp):
            raise ValueError("The input must be a pandas Timestamp object.")
        # Ensure timestamp is timezone-aware (if not already)
        timestamp = timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp
        tstz = timestamp.tz
        holiday_dates = self.USHolidays
        
        # General date-related features
        features_names = {
            "month",
            "day",
            "day_of_week",  # Monday=0, Sunday=6
            #"day_name": timestamp.day_name(),
            #"is_weekend": timestamp.dayofweek >= 5,  # True if Saturday or Sunday
            #"is_holiday": timestamp in country_holidays,
            #"holiday_name": country_holidays.get(timestamp, None),  # Name of the holiday if it's a holiday
            "quarter",
            "week_of_year",  # Week number of the year
            "is_month_start",
            "is_month_end",
            #"is_year_start": timestamp.is_year_start,
            #"is_year_end": timestamp.is_year_end,
            "days_to_next_holiday",
            "days_since_last_holiday",
            "season",  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
            "week_part",
        }
        # General date-related features
        features = [timestamp.month,
                    timestamp.day,
                    timestamp.dayofweek,  # Monday=0, Sunday=6
                    #"day_name": timestamp.day_name(),
                    #"is_weekend": timestamp.dayofweek >= 5,  # True if Saturday or Sunday
                    #"is_holiday": timestamp in country_holidays,
                    #"holiday_name": country_holidays.get(timestamp, None),  # Name of the holiday if it's a holiday
                    timestamp.quarter,
                    timestamp.isocalendar()[1],  # Week number of the year
                    timestamp.is_month_start,
                    timestamp.is_month_end,
                    #"is_year_start": timestamp.is_year_start,
                    #"is_year_end": timestamp.is_year_end,
                    np.min([(pd.Timestamp(h, tz=tstz) - timestamp).days for h in holiday_dates if pd.Timestamp(h, tz=tstz) >= timestamp]),
                    np.min([(timestamp - pd.Timestamp(h, tz=tstz)).days for h in holiday_dates if pd.Timestamp(h, tz=tstz) <= timestamp]),
                    timestamp.month % 12 // 3 + 1,  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
                    (0 if timestamp.dayofweek < 2 else 1 if timestamp.dayofweek < 4 else 2),
        ]
        return features
    
    def getFeaturesAndTarget(self, asset: AssetDataPolars, pricesArray: pl.Series, asset_taExt: pd.DataFrame, date: pd.Timestamp):
        aidx = DPl(asset.adjClosePrice).getNextLowerIndex(date)+1

        m = self.monthsHorizon
        numTimesteps = self.timesteps
        if (aidx - m * self.idxLengthOneMonth-1-numTimesteps)<0:
            print("Warning! Asset History does not span far enough.")
        if (aidx + self.daysAfterPrediction)>len(pricesArray):
            print("Warning! Future price does not exist in asset.")
            
        futurePrices = pricesArray.slice((aidx+self.daysAfterPrediction),1).to_numpy()
        futurePricesScaled = futurePrices/pricesArray.item(aidx)
        features = []
        features_timeseries = []
        for ts in range(1,numTimesteps+1):
            timestepIdx = numTimesteps - ts
            aidxTs = aidx - timestepIdx
            featuresTs = []
            
            pastPricesExt = pricesArray.slice(aidxTs - m*self.idxLengthOneMonth - 1, m * self.idxLengthOneMonth + 2).to_numpy()
            pastPrices = pastPricesExt[1:]
            pastPrices_log = np.log(pastPrices)
            pastPricesDiff = np.diff(pastPricesExt)
            pastPricesDiff = np.clip(pastPricesDiff, -1e3, 1e3)
            pastPricesDiff_exp = np.exp(pastPricesDiff)-1.0
            pastPricesDiff_exp = np.clip(pastPricesDiff_exp, -1e3, 1e3)
            pastReturns = pastPricesExt[1:] / pastPricesExt[:-1]
            pastReturns = np.clip(pastReturns, -1e3, 1e3)
            pastReturns_log = np.exp(pastReturns-1.0)
            pastReturns_log = np.clip(pastReturns_log, -1e3, 1e3)
            pastPricesScaled = pastPrices/pastPrices[-1]

            taRow = asset_taExt.iloc[aidxTs, :].values.tolist()
            country = pycountry.countries.lookup(asset.about.get('country','United States')).alpha_2

            #Mathematical Features
            mathFeatures = [pastPrices[-1], pastPrices_log[-1], pastPricesDiff[-1], pastPricesDiff_exp[-1], pastReturns[-1], pastReturns_log[-1]]
            featuresTs.extend(mathFeatures)

            #Fourier Features
            fourierFeatures = self.getFourierFeaturesFromPrice(pastPrices, 
                                                 multFactor=self.multFactor, 
                                                 fouriercutoff=self.fouriercutoff)
            featuresTs.extend(fourierFeatures)
            fourierFeatures_log = self.getFourierFeaturesFromPrice(pastPrices_log, 
                                                 multFactor=self.multFactor, 
                                                 fouriercutoff=self.fouriercutoff)
            featuresTs.extend(fourierFeatures_log)
            fourierDiffFeatures = self.getFourierFeaturesFromPrice([0] + pastPricesDiff + [0], 
                                                 multFactor=self.multFactor, 
                                                 fouriercutoff=self.fouriercutoff)
            featuresTs.extend(fourierDiffFeatures[1:]) #first element would be 0
            fourierDiffFeatures_exp = self.getFourierFeaturesFromPrice([0] + pastPricesDiff_exp + [0], 
                                                 multFactor=self.multFactor, 
                                                 fouriercutoff=self.fouriercutoff)
            featuresTs.extend(fourierDiffFeatures_exp[1:]) #first element would be 0
            fourierReturnFeatures = self.getFourierFeaturesFromPrice([1.0] + pastReturns + [1.0], 
                                                 multFactor=self.multFactor, 
                                                 fouriercutoff=self.fouriercutoff)
            featuresTs.extend(fourierReturnFeatures[1:]) #first element would be 0
            fourierReturnFeatures_log = self.getFourierFeaturesFromPrice([1.0] + pastReturns_log + [1.0], 
                                                 multFactor=self.multFactor, 
                                                 fouriercutoff=self.fouriercutoff)
            featuresTs.extend(fourierReturnFeatures_log[1:]) #first element would be 0

            #TA Features
            taFeatures = self.getTAFeatures(taRow)
            featuresTs.extend(taFeatures)

            #Categorical Informations
            catFeatures = self.getCategoryFeatures(asset)
            featuresTs.extend(catFeatures)

            #Seasonal Events Features
            seasFeatures = self.getSeasonalFeatures(date, country)
            featuresTs.extend(seasFeatures)
            
            features.extend(featuresTs)
            features_timeseries.append(featuresTs)
        
        target = self.getTargetFromPrice(futurePricesScaled-1, self.classificationInterval)
        target = target[0]

        return features, target, features_timeseries, futurePrices/pastPrices[-1]

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

        if self.trainStartDate == None \
             or self.trainEndDate == None \
             or self.testStartDate == None:
            raise ValueError("Data collection time is not defined.")
        
        self.USHolidays = self.__USHolidays()

        #Main Loop
        for ticker, asset in self.__assets.items():
            if asset.adjClosePrice is None or not 'AdjClose' in asset.adjClosePrice.columns:
                continue

            print(f"Processing asset: {asset.ticker}.  Processed {processedCounter} out of {len(self.__assets)}.")
            processedCounter += 1
            
            # Technical Analysis extension
            # This might lead to leakage if the ta process 'future' data
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
        self.y_val_timeseries = np.array(yvalPrice).astype(int)

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