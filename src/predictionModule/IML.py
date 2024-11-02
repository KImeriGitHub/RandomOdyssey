from abc import ABC, abstractmethod
import os
import numpy as np
import xgboost as xgb
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class IML(ABC):
    def __init__(self):
        self.XGBoostModel: xgb.XGBRegressor = xgb.XGBRegressor()
        self.CNNModel: Sequential = Sequential()
        self.X: np.array = np.array([])
        self.y: np.array = np.array([])

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
        if not fileName.lower().endswith('.h5'):
            fileName += '.h5'
        filePath = os.path.join(dirPath, f"{fileName}.")
        if self.CNNModel is None:
            raise ValueError('No model has been trained to save.')
        
        self.CNNModel.save(filePath)
        print("Model saved to mnist_mlp_model.h5")
        print(f'Model saved to {filePath}')

    def loadXGBoostModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.mdl'):
            fileName += '.mdl'
        filePath = os.path.join(dirPath, f"{fileName}.")
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Model file {filePath} does not exist.')
        
        # Initialize a new model and load the saved parameters
        self.XGBoostModel = xgb.XGBRegressor()
        self.XGBoostModel.load_model(filePath)
        print(f'Model loaded from {filePath}')

    def loadCNNModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.h5'):
            fileName += '.h5'
        filePath = os.path.join(dirPath, f"{fileName}.")
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Model file {filePath} does not exist.')
        
        # Load the model from the file
        self.CNNModel = load_model('mnist_mlp_model.h5')
        print("Model loaded successfully")