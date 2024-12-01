from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
import pickle
from typing import Dict
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler

from src.mathTools.RandomProjectionClassifier import RandomProjectionClassifier as rpc

class IML(ABC):
    def __init__(self):
        self.trainStartDate: pd.Timestamp = None
        self.trainEndDate: pd.Timestamp = None
        self.testStartDate: pd.Timestamp = None
        self.testEndDate: pd.Timestamp = None
        self.valStartDate: pd.Timestamp = None
        self.valEndDate: pd.Timestamp = None
        self.XGBoostModel: xgb.XGBClassifier = xgb.XGBClassifier()
        self.CNNModel: Sequential = Sequential()
        self.LSTMModel: Sequential = Sequential()
        self.LGBMModel: lgb.LGBMClassifier = lgb.LGBMClassifier()
        self.RPModel = rpc()
        self.X_train: np.array = np.array([])
        self.y_train: np.array = np.array([])
        self.X_test: np.array = np.array([])
        self.y_test: np.array = np.array([])
        self.X_val: np.array = np.array([])
        self.y_val: np.array = np.array([])
        self.X_train_timeseries: np.array = np.array([])
        self.y_train_timeseries: np.array = np.array([])
        self.X_test_timeseries: np.array = np.array([])
        self.y_test_timeseries: np.array = np.array([])
        self.X_val_timeseries: np.array = np.array([])
        self.y_val_timeseries: np.array = np.array([])
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.metadata: Dict = {}

        self.dataIsPrepared = False

    def saveXGBoostModel(self, dirPath: str, fileName:str):
        if not fileName.lower().endswith('.mdl'):
            fileName += '.mdl'
        filePath = os.path.join(dirPath, f"{fileName}")
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
        self.XGBoostModel = xgb.XGBClassifier()
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
            'X_val': self.X_val,
            'y_val': self.y_val,
            'X_train_timeseries': self.X_train_timeseries,
            'y_train_timeseries': self.y_train_timeseries,
            'X_test_timeseries': self.X_test_timeseries,
            'y_test_timeseries': self.y_test_timeseries,
            'X_val_timeseries': self.X_val_timeseries,
            'y_val_timeseries': self.y_val_timeseries,
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
        self.X_val = data.get('X_val', np.array([]))
        self.y_val = data.get('y_val', np.array([]))
        self.X_train_timeseries = data.get('X_train_timeseries', np.array([]))
        self.y_train_timeseries = data.get('y_train_timeseries', np.array([]))
        self.X_test_timeseries = data.get('X_test_timeseries', np.array([]))
        self.y_test_timeseries = data.get('y_test_timeseries', np.array([]))
        self.X_val_timeseries = data.get('X_val_timeseries', np.array([]))
        self.y_val_timeseries = data.get('y_val_timeseries', np.array([]))
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
    
    def traintestXGBModel(self, xgb_params=None, name_model_path:str = "", name_model_name: str = ""):
        if not self.dataIsPrepared:
            self.prepareData()

        # Split the data
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        X_val = self.X_val
        y_val = self.y_val

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
        
        if name_model_path != "" and name_model_name != "":
            self.saveXGBoostModel(name_model_path, name_model_name)

        # Make predictions
        if X_val.size != 0:
            y_pred = self.XGBoostModel.predict(X_val)
            y_pred_proba = self.XGBoostModel.predict_proba(X_val)
            
            test_acc = accuracy_score(y_val, y_pred)
            test_loss = log_loss(y_val, y_pred_proba)
        else:
            y_pred = self.XGBoostModel.predict(X_test)
            y_pred_proba = self.XGBoostModel.predict_proba(X_test)

            test_acc = accuracy_score(y_test, y_pred)
            test_loss = log_loss(y_test, y_pred_proba)

        self.metadata['XGBoostModel_accuracy'] = test_acc
        self.metadata['XGBoostModel_log_loss'] = test_loss
        print(f'\nTest accuracy: {test_acc:.4f}')
        print(f'Test log loss: {test_loss:.4f}')

    def traintestCNNModel(self, cnn_params=None, name_model_path:str = "", name_model_name: str = ""):
        if not self.dataIsPrepared:
            self.prepareData()
        
        # Split the data
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        X_val = self.X_val
        y_val = self.y_val
        
        num_classes = len(np.unique(self.y_train))

        y_cat_train = to_categorical(y_train)
        y_cat_test = to_categorical(y_test)
        y_cat_val = to_categorical(y_val)
        input_shape = X_train.shape[1]
        output_shape = y_cat_test.shape[1]

        # Define the CNN parameters
        if cnn_params == None:
            cnn_params = {
                'layers': [
                    {'units': 128, 'activation': 'relu'},
                    {'dropout': 0.2},
                    {'units': 64, 'activation': 'relu'},
                    {'units': num_classes, 'activation': ('softmax' if num_classes > 2 else 'sigmoid')}
                ],
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy',
                'metrics': ['accuracy'],
                'epochs': 20,
                'batch_size': 128
            }

        self.metadata['CNNModel_params'] = cnn_params
        
        # Define the model
        model = models.Sequential()
        for layer in cnn_params['layers']:
            if 'units' in layer and 'activation' in layer:
                model.add(layers.Dense(layer['units'], activation=layer['activation'], input_shape=(input_shape,)))
                input_shape = None  # Reset input shape after the first layer
            elif 'dropout' in layer:
                model.add(layers.Dropout(layer['dropout']))

        # Compile the model
        self.CNNModel.compile(optimizer=cnn_params['optimizer'],
                              loss=cnn_params['loss'],
                              metrics=cnn_params['metrics'])

        # Train the model
        history = self.CNNModel.fit(
            X_train, y_cat_train,
            epochs=cnn_params['epochs'],
            batch_size=cnn_params['batch_size'],
            verbose=2
        )
        
        if name_model_path != "" and name_model_name != "":
            self.saveCNNModel(name_model_path, name_model_name)
        
        if X_val.size != 0:
            test_loss, test_acc = self.CNNModel.evaluate(X_val, y_cat_val, verbose=0)
        else:
            test_loss, test_acc = self.CNNModel.evaluate(X_test, y_cat_test, verbose=0)
            
        self.metadata['CNNModel_accuracy'] = test_acc
        self.metadata['CNNModel_log_loss'] = test_loss
        print(f'\nTest accuracy: {test_acc:.4f}')
        print(f'Test log loss: {test_loss:.4f}')

    def traintestLGBMModel(self, lgbm_params=None, name_model_path:str = "", name_model_name: str = ""):
        if not self.dataIsPrepared:
            self.prepareData()

        # Split the data
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        X_val = self.X_val
        y_val = self.y_val

        # Define XGBoost parameters if not provided
        if lgbm_params is None:
            # Define LGBM parameters
            lgbm_params = {
                'n_estimators': 500,
                'learning_rate': 0.01,
                'max_depth': 8,
                'num_leaves': 32,
                'colsample_bytree': 0.1,
                'subsample': 0.8,
                'early_stopping_round': 100
            }
        self.metadata['LGBMModel_params'] = lgbm_params

        # Initialize and train LGBM model
        self.LGBMModel = lgb.LGBMClassifier(**lgbm_params)
        self.LGBMModel.fit(X_train, y_train,
                        eval_set=[(self.X_test, self.y_test)])
        
        if name_model_path != "" and name_model_name != "":
            self.saveLGBMModel(name_model_path, name_model_name)

        # Make predictions
        if X_val.size != 0:
            y_pred = self.LGBMModel.predict(X_val)
            y_pred_proba = self.LGBMModel.predict_proba(X_val)
            
            test_acc = accuracy_score(y_val, y_pred)
            test_loss = log_loss(y_val, y_pred_proba)
        else:
            y_pred = self.LGBMModel.predict(X_test)
            y_pred_proba = self.LGBMModel.predict_proba(X_test)

            test_acc = accuracy_score(y_test, y_pred)
            test_loss = log_loss(y_test, y_pred_proba)

        self.metadata['LGBMModel_accuracy'] = test_acc
        self.metadata['LGBMModel_log_loss'] = test_loss
        print(f'\nTest accuracy: {test_acc:.4f}')
        print(f'Test log loss: {test_loss:.4f}')

    def traintestRPModel(self, rp_params=None, name_model_path:str = "", name_model_name: str = ""):
        if not self.dataIsPrepared:
            self.prepareData()

        # Split the data
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        X_val = self.X_val
        y_val = self.y_val

        # Define XGBoost parameters if not provided
        if rp_params is None:
            # Define RPModel parameters
            rp_params = {
                'num_random_features': 2000,
                'regularization': 30,
                'max_iter': 10,
                'verbose': True,
                'random_state': None,
                'tuneReg_low_start': 1,
                'tuneReg_high_start': 100,
            }
        self.metadata['RPModel_params'] = rp_params

        # Initialize and train RPModel
        self.RPModel = rpc(**rp_params)

        # Train the model
        self.RPModel.fit(X_train, y_train)

        self.RPModel.tune_regularization(X_test, y_test, 
                                         low_start=rp_params['tuneReg_low_start'], 
                                         high_start=rp_params['tuneReg_high_start'], 
                                         max_iter=rp_params['max_iter'])
        
        if name_model_path != "" and name_model_name != "":
            self.saveRPModel(name_model_path, name_model_name)
        
        if X_val.size != 0:
            test_acc = self.RPModel.compute_accuracy(g=self.RPModel.g, Y_test = y_val, X_test=X_val)
        else:
            test_acc = self.RPModel.compute_accuracy(g=self.RPModel.g, Y_test = y_test, X_test=X_test)
            
        self.metadata['RPModel_g'] = self.RPModel.g
        self.metadata['RPModel_accuracy'] = test_acc
        print(f'\nBalance Parameter g: {self.RPModel.g:.4f}')
        print(f'\nTest accuracy: {test_acc:.4f}')
    
    def traintestLSTMModel(self, lstm_params=None, name_model_path:str = "", name_model_name: str = ""):
        if not self.dataIsPrepared:
            self.prepareData()
        
        # Split the data
        X_train = self.X_train_timeseries
        X_test = self.X_test_timeseries
        y_train = self.y_train_timeseries #shape (:,1)
        y_test = self.y_test_timeseries #shape (:,1)
        
        num_samples, timesteps, num_features = X_train.shape
        
        X_train_flat = X_train.reshape(num_samples, -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        X_train_scaled_flat = self.scaler_X.fit_transform(X_train_flat)
        X_test_scaled_flat = self.scaler_X.transform(X_test_flat)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        # Reshape the data to (num_samples, timesteps, features)
        X_train_scaled = X_train_scaled_flat.reshape(num_samples, timesteps, num_features)
        X_test_scaled = X_test_scaled_flat.reshape(X_test.shape[0], timesteps, num_features)  
        
        # Define LSTM parameters
        if lstm_params is None:
            lstm_params = {
                'units': 128,
                'dropout': 0.2,
                'dense_units': 64,
                'activation': 'relu',
                'optimizer': 'adam',
                'loss': 'mean_absolute_error',
                'metrics': ['mae'],
                'epochs': 20,
                'batch_size': 128,
                'early_stopping_round': 50
            }
        self.metadata['LSTMModel_params'] = lstm_params

        # Define callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=lstm_params['early_stopping_round'],
            restore_best_weights=True
        )
        
        # Initialize and compile LSTM model
        self.LSTMModel = models.Sequential([
            layers.LSTM(lstm_params['units'], input_shape=(timesteps, num_features)),
            layers.Dropout(lstm_params['dropout']),
            layers.Dense(lstm_params['dense_units'], activation = lstm_params['activation']),
            layers.Dense(1, activation='linear')
        ])

        self.LSTMModel.compile(optimizer=lstm_params['optimizer'],
                               loss=lstm_params['loss'],
                               metrics=lstm_params['metrics'])

        # Train the model
        history = self.LSTMModel.fit(
            X_train_scaled,
            y_train_scaled, 
            batch_size = lstm_params['batch_size'], 
            epochs = lstm_params['epochs'], 
            validation_data=(X_test_scaled, y_test_scaled),
            callbacks=[early_stop],
            verbose=1
        )
        
        if name_model_path != "" and name_model_name != "":
            self.saveLSTMModel(name_model_path, name_model_name)

        # Evaluate the model
        test_loss, test_mae  = self.LSTMModel.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        self.metadata['LSTMModel_loss'] = test_loss
        self.metadata['LSTMModel_mae'] = test_mae
        print(f'\nTest MAE: {test_mae:.4f}')
        print(f'Test Loss (MSE): {test_loss:.4f}')