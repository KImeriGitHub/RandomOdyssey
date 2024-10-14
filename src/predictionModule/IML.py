from abc import ABC, abstractmethod
import os
import numpy as np
import xgboost as xgb

class IML(ABC):
    def __init__(self):
        self.model: xgb.XGBRegressor = xgb.XGBRegressor()
        self.X: np.array = np.array([])
        self.y: np.array = np.array([])

    def saveModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.mdl'):
            fileName += '.mdl'
        filePath = os.path.join(dirPath, f"{fileName}.")
        if self.model is None:
            raise ValueError('No model has been trained to save.')
        
        # Save the model to the specified filepath
        self.model.save_model(filePath)
        print(f'Model saved to {filePath}')

    def loadModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.mdl'):
            fileName += '.mdl'
        filePath = os.path.join(dirPath, f"{fileName}.")
        if not os.path.exists(filePath):
            raise FileNotFoundError(f'Model file {filePath} does not exist.')
        
        # Initialize a new model and load the saved parameters
        self.model = xgb.XGBRegressor()
        self.model.load_model(filePath)
        print(f'Model loaded from {filePath}')