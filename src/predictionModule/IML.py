from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM
from src.mathTools.RandomProjectionClassifier import RandomProjectionClassifier as rpc
import lightgbm as lgb
import joblib
import pickle

class IML(ABC):
    def __init__(self):
        self.trainStartDate: pd.Timestamp = None,
        self.trainEndDate: pd.Timestamp = None,
        self.testStartDate: pd.Timestamp = None,
        self.testEndDate: pd.Timestamp = None,
        self.valStartDate: pd.Timestamp = None,
        self.valEndDate: pd.Timestamp = None,
        self.XGBoostModel: xgb.XGBClassifier = xgb.XGBClassifier()
        self.CNNModel: Sequential = Sequential()
        self.LSTMModel: Sequential = Sequential()
        self.LGBMModel = lgb.LGBMRegressor()
        self.RPModel = rpc()
        self.X_train: np.array = np.array([])
        self.y_train: np.array = np.array([])
        self.X_test: np.array = np.array([])
        self.y_test: np.array = np.array([])
        self.X_train_timeseries: np.array = np.array([])
        self.y_train_timeseries: np.array = np.array([])
        self.X_test_timeseries: np.array = np.array([])
        self.y_test_timeseries: np.array = np.array([])
        self.metadata: Dict = {}

        self.dataIsPrepared = False

    def saveXGBoostModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.mdl'):
            fileName += '.mdl'
        filePath = os.path.join(dirPath, f"{fileName}.")
        if self.XGBoostModel is None:
            raise ValueError('No model has been trained to save.')
        
        # Save the model to the specified filepath
        self.XGBoostModel.save_model(filePath)
        print(f'Model saved to {filePath}')

    def saveCNNModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.keras'):
            fileName += '.keras'
        filePath = os.path.join(dirPath, f"{fileName}")
        if self.CNNModel is None:
            raise ValueError('No model has been trained to save.')
        
        self.CNNModel.save(filePath, overwrite=True)
        print(f'Model saved to {filePath}')

    def loadXGBoostModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.mdl'):
            fileName += '.mdl'
        filePath = os.path.join(dirPath, f"{fileName}")
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Model file {filePath} does not exist.')
        
        # Initialize a new model and load the saved parameters
        self.XGBoostModel = xgb.XGBRegressor()
        self.XGBoostModel.load_model(filePath)
        print(f'Model loaded from {filePath}')

    def loadCNNModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.keras'):
            fileName += '.keras'
        filePath = os.path.join(dirPath, f"{fileName}")
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Model file {filePath} does not exist.')
        
        # Load the model from the file
        self.CNNModel = load_model(filePath)
        print("Model loaded successfully")

    def saveLSTMModel(self, dirPath: str, fileName: str):
        if not fileName.lower().endswith('.keras'):
            fileName += '.keras'
        filePath = os.path.join(dirPath, fileName)
        if self.LSTMModel is None:
            raise ValueError('No model has been trained to save.')
        self.LSTMModel.save(filePath, overwrite=True)
        print(f'Model saved to {filePath}')

    def loadLSTMModel(self, dirPath: str, fileName: str):
        if not fileName.lower().endswith('.keras'):
            fileName += '.keras'
        filePath = os.path.join(dirPath, fileName)
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Model file {filePath} does not exist.')
        self.LSTMModel = load_model(filePath)
        print(f'Model loaded from {filePath}')

    def saveLGBMModel(self, dirPath: str, fileName: str):
        if not fileName.lower().endswith('.pkl'):
            fileName += '.pkl'
        filePath = os.path.join(dirPath, fileName)
        if self.LGBMModel is None:
            raise ValueError('No model has been trained to save.')
        joblib.dump(self.LGBMModel, filePath)
        print(f'Model saved to {filePath}')

    def loadLGBMModel(self, dirPath: str, fileName: str):
        if not fileName.lower().endswith('.pkl'):
            fileName += '.pkl'
        filePath = os.path.join(dirPath, fileName)
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Model file {filePath} does not exist.')
        self.LGBMModel = joblib.load(filePath)
        print(f'Model loaded from {filePath}')

    def saveRPModel(self, dirPath: str, fileName: str):
        if not fileName.lower().endswith('.pkl'):
            fileName += '.pkl'
        filePath = os.path.join(dirPath, fileName)
        if self.RPModel is None:
            raise ValueError('No model has been trained to save.')
        joblib.dump(self.RPModel, filePath)
        print(f'Model saved to {filePath}')

    def loadRPModel(self, dirPath: str, fileName: str):
        if not fileName.lower().endswith('.pkl'):
            fileName += '.pkl'
        filePath = os.path.join(dirPath, fileName)
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Model file {filePath} does not exist.')
        self.RPModel = joblib.load(filePath)
        print(f'Model loaded from {filePath}')

    def save_data(self, dirPath: str, fileName: str):
        if not fileName.lower().endswith('.pkl'):
            fileName += '.pkl'
        filePath = os.path.join(dirPath, fileName)
        data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'X_train_timeseries': self.X_train_timeseries,
            'y_train_timeseries': self.y_train_timeseries,
            'X_test_timeseries': self.X_test_timeseries,
            'y_test_timeseries': self.y_test_timeseries,
            'trainStartDate': self.trainStartDate,
            'trainEndDate': self.trainEndDate,
            'testStartDate': self.testStartDate,
            'testEndDate': self.testEndDate,
            'valStartDate': self.valStartDate,
            'valEndDate': self.valEndDate,
            'metadata': self.metadata
        }
        with open(filePath, 'wb') as f:
            pickle.dump(data, f)
        print(f'Data and metadata saved to {filePath}')

    def load_data(self, dirPath: str, fileName: str):
        if not fileName.lower().endswith('.pkl'):
            fileName += '.pkl'
        filePath = os.path.join(dirPath, fileName)
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Data file {filePath} does not exist.')
        with open(filePath, 'rb') as f:
            data = pickle.load(f)
        self.X_train = data.get('X_train', np.array([]))
        self.y_train = data.get('y_train', np.array([]))
        self.X_test = data.get('X_test', np.array([]))
        self.y_test = data.get('y_test', np.array([]))
        self.X_train_timeseries = data.get('X_train_timeseries', np.array([]))
        self.y_train_timeseries = data.get('y_train_timeseries', np.array([]))
        self.X_test_timeseries = data.get('X_test_timeseries', np.array([]))
        self.y_test_timeseries = data.get('y_test_timeseries', np.array([]))
        self.trainStartDate = data.get('trainStartDate', None)
        self.trainEndDate = data.get('trainEndDate', None)
        self.testStartDate = data.get('testStartDate', None)
        self.testEndDate = data.get('testEndDate', None)
        self.valStartDate = data.get('valStartDate', None)
        self.valEndDate = data.get('valEndDate', None)
        self.metadata = data.get('metadata', {})
        print(f'Data and metadata loaded from {filePath}')
        self.dataIsPrepared = True
        return self.metadata