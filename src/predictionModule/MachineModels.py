import os
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import logging
import datetime
import re
import time
from scipy import stats

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
import time

logger = logging.getLogger(__name__)

class MachineModels:
    """
        Base class for machine learning models used in time series prediction.
        Provides methods for training, evaluating, and predicting with various models.
    """
    
    default_params = {
        "LSTM_units": 32,
        "LSTM_num_layers": 3,
        "LSTM_dropout": 0.001,
        "LSTM_recurrent_dropout": 0.001,
        "LSTM_learning_rate": 0.001,
        "LSTM_optimizer": "adam",
        "LSTM_bidirectional": True,
        "LSTM_batch_size": 2**10,
        "LSTM_epochs": 10,
        "LSTM_l1": 0.001,
        "LSTM_l2": 0.001,
        "LSTM_inter_dropout": 0.001,
        "LSTM_input_gaussian_noise": 0.001,
        "LSTM_conv1d": True,
        "LSTM_conv1d_kernel_size": 3,
        "LSTM_loss": "mse",
        
        "LGB_num_boost_round": 1500,
        "LGB_lambda_l1": 0.04614242128149404,
        "LGB_lambda_l2": 0.009786276249261908,
        "LGB_feature_fraction": 0.20813359498274574,
        "LGB_num_leaves": 182,
        "LGB_max_depth": 11,
        "LGB_learning_rate": 0.02887444408582576,
        "LGB_min_data_in_leaf": 272,
        "LGB_min_gain_to_split": 0.10066457576238419,
        "LGB_path_smooth": 0.5935679203578974,
        "LGB_min_sum_hessian_in_leaf": 0.3732876155751053,
        "LGB_max_bin": 150,
    }
    
    def __init__(self, params: dict):
        self.params = {**self.default_params, **params}
        
    def run_LGB(self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        weights: np.ndarray = None,
    ) -> tuple[lgb.Booster, dict]:
        
        num_boost_round = self.params['LGB_num_boost_round']
        lgb_params  = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'objective': 'regression',
            #'alpha': 0.85,
            'metric': 'l2_root',  # NOTE: the string 'rsme' is not recognized, v 4.5.0
            'lambda_l1': self.params['LGB_lambda_l1'],
            'lambda_l2': self.params['LGB_lambda_l2'],
            'early_stopping_rounds': num_boost_round//10 ,
            'feature_fraction': self.params['LGB_feature_fraction'],
            'num_leaves': self.params['LGB_num_leaves'], 
            'max_depth': self.params['LGB_max_depth'],
            'learning_rate': self.params['LGB_learning_rate'],
            'min_data_in_leaf': self.params['LGB_min_data_in_leaf'],
            'min_gain_to_split': self.params['LGB_min_gain_to_split'],
            'path_smooth': self.params['LGB_path_smooth'],
            'min_sum_hessian_in_leaf': self.params['LGB_min_sum_hessian_in_leaf'],
            'random_state': 41,
        }   

        if weights is None:
            weights = np.ones_like(y_train, dtype=np.float32)
        train_data = lgb.Dataset(X_train, label = y_train, weight=weights)
        if y_test is None or np.any(np.isnan(y_test)):
            test_data = None
        else:
            test_data = lgb.Dataset(X_test, label = y_test, reference=train_data)

        def print_eval_after_100(env):
            if env.iteration % 100 == 0 or env.iteration == num_boost_round:
                results = [
                    f"{data_name}'s {eval_name}: {result}"
                    for data_name, eval_name, result, _ in env.evaluation_result_list
                ]
                logger.info(f"Iteration {env.iteration}: " + ", ".join(results))

        #lgb_params['metric'] = lgb_params['metric'] if test_data is not None else None
        lgb_params['early_stopping_rounds'] = lgb_params['early_stopping_rounds'] if test_data is not None else None
        gbm = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[test_data] if test_data is not None else None,
            num_boost_round=num_boost_round,
            callbacks=[print_eval_after_100] #if test_data is not None else None
        )

        res_dict = {
            'feature_importance': gbm.feature_importance(importance_type='gain'),
        }
        if test_data is not None:
            res_dict.update({
                'best_iteration': gbm.best_iteration,
                'best_score': gbm.best_score['valid_0']['rmse']
            })

        return gbm, res_dict
    
    def __run_LGB_lambdarank(self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        weights: np.ndarray = None,
    ) -> tuple[lgb.Booster, dict]:
        # Must be adjusted to the class
        pass
        """
        num_boost_round = self.params['TreeTime_lgb_num_boost_round']
        lgb_params  = {
            'verbosity': -1,
            'n_jobs': -1,
            'is_unbalance': True,
            'objective': 'lambdarank',
            'ndcg_eval_at': [1,5],
            'label_gain': tuple(np.array(range(1,400))**(2)), 
            'metric': 'ndcg',  
            #'lambdarank_truncation_level': 10,
            #'lambdarank_norm': True,
            'lambda_l1': self.params['TreeTime_lgb_lambda_l1'],
            'lambda_l2': self.params['TreeTime_lgb_lambda_l2'],
            'early_stopping_rounds': num_boost_round//2,
            'feature_fraction': self.params['TreeTime_lgb_feature_fraction'],
            'num_leaves': self.params['TreeTime_lgb_num_leaves'], 
            'max_depth': self.params['TreeTime_lgb_max_depth'],
            'learning_rate': self.params['TreeTime_lgb_learning_rate'],
            'min_data_in_leaf': self.params['TreeTime_lgb_min_data_in_leaf'],
            'min_gain_to_split': self.params['TreeTime_lgb_min_gain_to_split'],
            'path_smooth': self.params['TreeTime_lgb_path_smooth'],
            'min_sum_hessian_in_leaf': self.params['TreeTime_lgb_min_sum_hessian_in_leaf'],
            "max_bin": self.params["TreeTime_lgb_max_bin"],
            'random_state': 41,
        }   
        
        counts_df = self.meta_pl_train.group_by("date", maintain_order=True).agg(pl.count("date").alias("cnt"))
        group_sizes = counts_df['cnt'].to_list()
        
        test_size_pct = self.params['TreeTime_lgb_test_size_pct']   
        n_test = max(1, int(len(group_sizes) * test_size_pct))
        train_group_sizes = group_sizes[:-n_test]
        test_group_sizes  = group_sizes[-n_test:]
        split_idx = sum(train_group_sizes)
        
        ytree_df = pl.DataFrame({
            "date": self.meta_pl_train.select('date').to_series(), 
            "y": self.train_ytree
        }).with_columns(
            pl.col("y")
            .rank(method="dense")
            .over("date")
            .cast(pl.UInt64)  
            .alias("y_rank")
        )
        
        y_rank_list = ytree_df["y_rank"].to_list()
        # split features and labels
        X_all = self.train_Xtree
        y_all = y_rank_list
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]
        train_data = lgb.Dataset(X_train, label=y_train, group=train_group_sizes)
        valid_data = lgb.Dataset(X_test,  label=y_test,  group=test_group_sizes)
        
        def print_eval_after_100(env):
            if env.iteration % 100 == 0 or env.iteration == num_boost_round:
                results = [
                    f"{data_name}'s {eval_name}: {result}"
                    for data_name, eval_name, result, _ in env.evaluation_result_list
                ]
                logger.info(f"Iteration {env.iteration}: " + ", ".join(results))
        
        evals_result = {}
        gbm: lgb.Booster = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'test'],
            callbacks=[
                #record_evaluation(evals_result),
                print_eval_after_100
            ],
            num_boost_round=num_boost_round,
        )
        #ndcg5 = np.array(evals_result['test']['ndcg@5'])
        #best_it = int(ndcg5.argmax()) + 1
        #best_score = ndcg5.max()
        #gbm.best_iteration = best_it
        #gbm.best_score['test']['ndcg@5'] = best_score
        
        return gbm
        """
        
    def run_LSTM_tf(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_test: np.ndarray = None,
            y_test: np.ndarray = None
        ) -> tuple[tf.keras.Model, dict]:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, GaussianNoise, LSTM, Bidirectional, Dropout, Dense, Conv1D
        from tensorflow.keras import regularizers, backend as K
        from tensorflow.keras.optimizers import Adam, RMSprop
        from tensorflow.keras.losses import MeanSquaredError
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
        from tensorflow.keras.metrics import RootMeanSquaredError, R2Score
        
        # Hyperparameters to tune
        lstm_units = self.params["LSTM_units"]
        num_layers = self.params["LSTM_num_layers"]
        dropout = self.params.get("LSTM_dropout", 0.001)
        recurrent_dropout = self.params["LSTM_recurrent_dropout"]
        learning_rate = self.params["LSTM_learning_rate"]
        optimizer_name = self.params["LSTM_optimizer"]
        bidirectional = self.params["LSTM_bidirectional"]
        batch_size = self.params["LSTM_batch_size"]
        epochs = self.params["LSTM_epochs"]
        loss_name = self.params["LSTM_loss"]

        # Regularization hyperparameters
        l1 = self.params.get("LSTM_l1", 0.0)
        l2 = self.params.get("LSTM_l2", 0.0)
        inter_dropout = self.params.get("LSTM_inter_dropout", 0.0)
        noise_std = self.params.get("LSTM_input_gaussian_noise", 0.0)

        # Conv1D option
        use_conv1d = self.params.get("LSTM_conv1d", False)
        conv_filters = lstm_units
        conv_kernel = self.params.get("LSTM_conv1d_kernel_size", 3)

        # Build model
        model = Sequential([Input(shape=X_train.shape[1:])])
        # Add Gaussian noise to inputs
        if noise_std > 0:
            model.add(GaussianNoise(noise_std))
        # Add Conv1D layer if opted in
        if use_conv1d:
            model.add(Conv1D(filters=conv_filters,
                            kernel_size=conv_kernel,
                            padding='same',
                            activation='relu'))
        # Add LSTM layers with regularization and dropout
        for i in range(num_layers):
            return_seq = i < (num_layers - 1)
            lstm_layer = LSTM(
                lstm_units,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)
            )
            if bidirectional:
                model.add(Bidirectional(lstm_layer))
            else:
                model.add(lstm_layer)
            # Add dropout between layers
            if inter_dropout > 0 and return_seq:
                model.add(Dropout(inter_dropout))
        # Output layer
        model.add(Dense(1, activation='relu', kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)))
        
        # Optimizer
        if optimizer_name == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Loss function
        def quantile_loss(q):
            def loss(y_true, y_pred):
                e = y_true - y_pred
                return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))
            return loss
        def r2_keras(y_true, y_pred):
            """
            Returns R^2 metric: 1 - SS_res / SS_tot
            """
            ss_res =  K.sum(K.square(y_true - y_pred)) 
            ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
            # avoid division by zero
            return 1 - ss_res/(ss_tot + K.epsilon())
        def neg_r2_loss(y_true, y_pred):
            """
            Loss function to *maximize* R^2 by minimizing its negative.
            """
            return -r2_keras(y_true, y_pred)
        if loss_name == "mse":
            loss_lstm = MeanSquaredError()
        elif loss_name == "r2":
            loss_lstm = neg_r2_loss
        else:
            # handles quantile_1,3,5,7,9 etc.
            q = int(loss_name.split("_")[1]) / 10.0
            loss_lstm = quantile_loss(q)
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss_lstm,
            metrics=[MeanSquaredError(name='mse'),
                    RootMeanSquaredError(name='rmse')]
        )
        
        # Callbacks
        class TimeLimit(Callback):
            def __init__(self, max_seconds): super().__init__(); self.max_seconds = max_seconds
            def on_train_begin(self, logs=None): self.t0 = time.time()
            def on_batch_end(self, batch, logs=None): (time.time() - self.t0 > self.max_seconds) and setattr(self.model, 'stop_training', True)
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)
        time_cb = TimeLimit(3600) 
        # Train
        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[es, rlrop, time_cb],
            shuffle=False,
        )

        # Log metrics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_rmse = history.history['rmse'][-1]
        final_val_rmse = history.history['val_rmse'][-1]

        logger.info(f"  Train    -> loss: {final_loss:.4f},  rmse: {final_rmse:.4f}")
        logger.info(f"  Validate -> loss: {final_val_loss:.4f}, rmse: {final_val_rmse:.4f}")
        
        model.summary(
            print_fn=lambda line: logger.info(
                re.sub(r'[\u2500-\u257F]+', '', line)
            )
        )

        return model, history.history    
    
    def predict_LSTM_tf(self,
            model: tf.keras.Model, 
            X: np.ndarray,
            batch_size: int = 2**10
        ) -> np.ndarray:
        return model.predict(X, batch_size=batch_size)[:,0]
        
    def run_LSTM_torch(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_test: np.ndarray = None,
            y_test: np.ndarray = None, 
            device='cpu'
        ) -> tuple[torch.nn.Module, dict]:
        # Hyperparameters
        lstm_units = self.params['LSTM_units']
        num_layers = self.params['LSTM_num_layers']
        dropout = self.params['LSTM_dropout']
        recurrent_dropout = self.params['LSTM_recurrent_dropout']
        learning_rate = self.params['LSTM_learning_rate']
        optimizer_name = self.params['LSTM_optimizer']
        bidirectional = self.params['LSTM_bidirectional']
        batch_size = self.params['LSTM_batch_size']
        epochs = self.params['LSTM_epochs']
        loss_name = self.params['LSTM_loss']
        l1 = self.params.get('LSTM_l1', 0.0)
        l2 = self.params.get('LSTM_l2', 0.0)
        inter_dropout = self.params.get('LSTM_inter_dropout', 0.0)
        noise_std = self.params.get('LSTM_input_gaussian_noise', 0.0)
        use_conv1d = self.params.get('LSTM_conv1d', False)
        conv_kernel = self.params.get('LSTM_conv1d_kernel_size', 3)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Model
        model = LSTM_Torch(
            input_size=X_train.shape[-1],
            lstm_units=lstm_units,
            num_layers=num_layers,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            bidirectional=bidirectional,
            l1=l1,
            l2=l2,
            use_conv1d=use_conv1d,
            conv_kernel=conv_kernel,
            noise_std=noise_std,
            inter_dropout=inter_dropout
        ).to(device)

        # Loss functions
        def quantile_loss(q):
            def loss_fn(y_pred, y_true):
                e = y_true - y_pred
                return torch.mean(torch.max(q * e, (q - 1) * e))
            return loss_fn

        def r2_metric(y_pred, y_true):
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            return 1 - ss_res / (ss_tot + 1e-6)

        def neg_r2_loss(y_pred, y_true):
            return -r2_metric(y_pred, y_true)

        # Loss & optimizer
        if loss_name == 'mse':
            criterion = nn.MSELoss()
        elif loss_name == 'r2':
            criterion = lambda pred, true: neg_r2_loss(pred, true)
        else:
            q = int(loss_name.split('_')[1]) / 10.0
            criterion = quantile_loss(q)
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=l2
        )
        if optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(), lr=learning_rate, weight_decay=l2
            )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=2
        )

        best_rmse, wait = float('inf'), 0
        start_time = time.time()

        for epoch in trange(epochs, desc='Epochs'):
            model.train()
            sum_sq_error = 0.0
            total_samples = 0

            for X_batch, y_batch in tqdm(train_loader, desc='Training', leave=False):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                preds = model(X_batch).squeeze()
                # compute loss (with optional L1)
                loss = criterion(preds, y_batch)
                if l1 > 0:
                    loss = loss + l1 * sum(p.abs().sum() for p in model.parameters())
                loss.backward()
                optimizer.step()

                # accumulate squared error for RMSE
                # note: detach so it doesn't track grads
                se = ((preds.detach() - y_batch) ** 2).sum().item()
                sum_sq_error += se
                total_samples += y_batch.numel()

                if time.time() - start_time > 3600:
                    break

            train_rmse = (sum_sq_error / total_samples) ** 0.5

            # validation as before
            model.eval()
            val_rmses = []
            with torch.no_grad():
                for X_batch, y_batch in tqdm(val_loader, desc='Validation', leave=False):
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch).squeeze()
                    mse = nn.MSELoss()(preds, y_batch)
                    val_rmses.append(torch.sqrt(mse).item())
            val_rmse = sum(val_rmses) / len(val_rmses)

            logger.info(f"Epoch {epoch+1}/{epochs} — "
                f"Train RMSE: {train_rmse:.4f} — "
                f"Validation RMSE: {val_rmse:.4f}")

            scheduler.step(val_rmse)
            
            if val_rmse < best_rmse:
                best_rmse, wait = val_rmse, 0
                best_state = model.state_dict()
            else:
                wait += 1
                if wait >= 3:
                    break
            if time.time() - start_time > 3600:
                break

        model.load_state_dict(best_state)
        return model, {'val_rmse': best_rmse, 'history': None}
    
    def predict_LSTM_torch(self,
            model: torch.nn.Module,
            X: np.ndarray,
            batch_size: int = 2**10,
            device: str = 'cpu'
        ) -> np.ndarray:
        """
        Predict using a trained LSTM model.
        
        Args:
            model (torch.nn.Module): The trained LSTM model.
            X (np.ndarray): Input data for prediction.
            batch_size (int): Batch size for prediction.
            device (str): Device to run the model on ('cpu' or 'cuda').
        
        Returns:
            np.ndarray: Predicted values.
        """
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch[0].to(device)
                preds = model(X_batch).squeeze().cpu().numpy()
                predictions.append(preds)
        
        return np.concatenate(predictions, axis=0)

class LSTM_Torch(nn.Module):
    def __init__(self, 
            input_size,
            lstm_units,
            num_layers,
            dropout,
            recurrent_dropout,
            bidirectional,
            l1=0.0,
            l2=0.0,
            use_conv1d=False,
            conv_kernel=3,
            noise_std=0.0,
            inter_dropout=0.0
        ):
        super().__init__()
        self.use_conv1d = use_conv1d
        self.noise_std = noise_std
        self.inter_dropout = inter_dropout
        if use_conv1d:
            self.conv1d = nn.Conv1d(
                in_channels=input_size,
                out_channels=lstm_units,
                kernel_size=conv_kernel,
                padding=conv_kernel//2
            )
            input_size = lstm_units
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(inter_dropout) if inter_dropout > 0 else None
        self.output = nn.Linear(
            lstm_units * (2 if bidirectional else 1),
            1
        )
        self.l1 = l1
        self.l2 = l2
    def forward(self, x):
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.use_conv1d:
            x = x.transpose(1, 2)
            x = self.conv1d(x)
            x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out_last = out[:, -1, :]
        if self.dropout:
            out_last = self.dropout(out_last)
        return self.output(out_last)
