import os
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from lightgbm.callback import record_evaluation
import logging
import datetime
import re
import time
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GaussianNoise, LSTM, Bidirectional, Dropout, Dense, Conv1D
from tensorflow.keras import regularizers, backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.metrics import RootMeanSquaredError, R2Score

from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.mathTools.DistributionTools import DistributionTools

logger = logging.getLogger(__name__)

class TreeTimeML:
    # Class-level default parameters
    DEFAULT_PARAMS = {
        "daysAfterPrediction": 10,
        'timesteps': 20,
    }

    def __init__(
            self, 
            train_start_date: datetime.date,
            test_dates: list[datetime.date],
            group: str,
            params: dict = None,
        ):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.group = group
        self.train_start_date = train_start_date
        self.test_dates = test_dates
        
        self.min_test_date = min(self.test_dates)
        self.max_test_date = max(self.test_dates)
        
        # Assign parameters to instance variables
        self.daysAfter = self.params['daysAfterPrediction']
        self.timesteps = self.params['timesteps'] 
        
        self.featureTreeNames: list
        self.featureTimeNames: list
        self.meta_pl_train: pl.DataFrame
        self.meta_pl_test: pl.DataFrame
        self.test_Xtree: np.array
        self.test_Xtime: np.array
        self.test_ytree: np.array
        self.test_ytime: np.array
        self.train_Xtree: np.array
        self.train_Xtime: np.array
        self.train_ytree: np.array
        self.train_ytime: np.array
        
    def load_and_filter_sets(self, main_path: str = "src/featureAlchemy/bin/"):
        years = np.arange(self.train_start_date.year, self.max_test_date.year + 1)
        meta_all_pl_list = []
        namestree: list = []
        namestime: list = []
        all_Xtree_list = []
        all_Xtime_list = []
        mask_list = []
        
        # check whether files are there otherwise make error
        for year in years:
            label = str(year)
            if not os.path.isfile(main_path + f"TreeFeatures_{label}_{self.group}.npz"):
                raise FileNotFoundError(f"Tree feature sets not found.")
            if not os.path.isfile(main_path + f"TimeFeatures_{label}_{self.group}.npz"):
                raise FileNotFoundError(f"Time feature sets not found.")
            
        testdate_in_db = np.zeros(len(self.test_dates), dtype=bool)
        for year in years:
            label = str(year)
            
            tree_npz = main_path +f"TreeFeatures_{label}_{self.group}.npz"
            time_npz = main_path +f"TimeFeatures_{label}_{self.group}.npz"
            data_tree = np.load(tree_npz, allow_pickle=True)
            data_time = np.load(time_npz, allow_pickle=True)
            meta_tree  = data_tree['meta_tree']
            feats_tree = data_tree['treeFeatures']
            names_tree = data_tree['treeFeaturesNames']
            #meta_time  = data_time['meta_time']
            feats_time = data_time['timeFeatures']
            names_time = data_time['timeFeaturesNames']
            
            for idx, test_date in enumerate(self.test_dates):
                if test_date in meta_tree["date"]:
                    testdate_in_db[idx] = True
            
            # shorten to timesteps
            if feats_time.shape[1] > self.timesteps:
                feats_time = feats_time[:,-self.timesteps:,:]
            
            mask = self.__forge_filtermask(feats_tree, names_tree)
            
            # apply masks
            meta_pl_all_loop: pl.DataFrame = pl.DataFrame({
                "date":   meta_tree["date"],
                "ticker": meta_tree["ticker"],
                "Close":  meta_tree["Close"],
            })
            
            namestree = names_tree
            namestime = names_time
            mask_list.append(mask)
            meta_all_pl_list.append(meta_pl_all_loop)
            all_Xtree_list.append(feats_tree[mask])
            all_Xtime_list.append(feats_time[mask])

        mask_all = np.concatenate(mask_list)
        meta_pl_all: pl.DataFrame = pl.concat(meta_all_pl_list)
        self.featureTreeNames = namestree
        self.featureTimeNames = namestime
        all_Xtree_pre = np.concatenate(all_Xtree_list, axis=0)
        all_Xtime_pre = np.concatenate(all_Xtime_list, axis=0)  
        
        for idx, test_date in enumerate(self.test_dates):
            if not testdate_in_db[idx]:
                logger.warning(f"Test date {test_date} not found in the database. Resetting to last trading day.")
                self.test_dates[idx] = meta_pl_all.filter(pl.col("date") <= test_date).select("date").max()["date"].item()
        
        # get target value
        meta_pl_all = self.__add_target(meta_pl_all, self.daysAfter)
        meta_pl_all = meta_pl_all.with_columns((pl.col("target_close")  / pl.col("Close")).alias("target_ratio"))
        tar_all = meta_pl_all["target_close"].to_numpy().flatten()
        cur_all = meta_pl_all["Close"].to_numpy().flatten()
        tar_masked = tar_all[mask_all]
        cur_masked = cur_all[mask_all]
        meta_pl = meta_pl_all.filter(mask_all)
        
        # Filter start date end date
        mask_at_test_dates = (meta_pl["date"].is_in(self.test_dates)).to_numpy()
        mask_inbetween_date = ((meta_pl["date"] >= self.train_start_date) & (meta_pl["date"] <= (self.min_test_date - pd.Timedelta(days=self.daysAfter)))).to_numpy()
        
        rat_at_test_date = tar_masked[mask_at_test_dates] / cur_masked[mask_at_test_dates]
        rat_inbetween = tar_masked[mask_inbetween_date] / cur_masked[mask_inbetween_date]
        
        self.meta_pl_train = meta_pl.filter(mask_inbetween_date)
        self.meta_pl_test  = meta_pl.filter(mask_at_test_dates)
        
        self.test_Xtree = all_Xtree_pre[mask_at_test_dates]
        self.test_Xtime = all_Xtime_pre[mask_at_test_dates]
        
        self.test_ytree = np.clip(rat_at_test_date, 0.5, 1.5)
        self.test_ytime = np.clip((rat_at_test_date - 1.0) * 5, -0.5, 0.5) + 0.5
        
        self.train_Xtree = all_Xtree_pre[mask_inbetween_date]
        self.train_Xtime = all_Xtime_pre[mask_inbetween_date]
        
        self.train_ytree = np.clip(rat_inbetween, 0.5, 1.5)
        self.train_ytime = np.clip((rat_inbetween - 1.0) * 5, -0.5, 0.5) + 0.5
        
        #  Remove outsider and nan
        mask_outsiders_at_test = (self.test_ytree > 1.30) | (self.test_ytree < 0.70)
        mask_outsiders_inbetween = (self.train_ytree > 1.30) | (self.train_ytree < 0.70)
        mask_nan_inbetween_tree = np.isnan(self.train_ytree) 
        mask_nan_inbetween_time = np.isnan(self.train_ytime)
        
        if any(np.isnan(self.test_ytree)):
            logger.warning("NaN values found in test_ytree. Removing them.")
        if any(np.isnan(self.test_ytime)):
            logger.warning("NaN values found in test_ytime. Removing them.")
        
        self.test_Xtree = self.test_Xtree[~mask_outsiders_at_test]
        self.test_Xtime = self.test_Xtime[~mask_outsiders_at_test]
        self.test_ytree = self.test_ytree[~mask_outsiders_at_test]
        self.test_ytime = self.test_ytime[~mask_outsiders_at_test]
        self.train_Xtree = self.train_Xtree[~mask_outsiders_inbetween & ~mask_nan_inbetween_tree]
        self.train_Xtime = self.train_Xtime[~mask_outsiders_inbetween & ~mask_nan_inbetween_time]
        self.train_ytree = self.train_ytree[~mask_outsiders_inbetween & ~mask_nan_inbetween_tree]
        self.train_ytime = self.train_ytime[~mask_outsiders_inbetween & ~mask_nan_inbetween_time]
        self.meta_pl_train = self.meta_pl_train.filter(~mask_outsiders_inbetween & ~mask_nan_inbetween_tree)
        self.meta_pl_test = self.meta_pl_test.filter(~mask_outsiders_at_test)
        
        # Scale Tree features
        scaler = StandardScaler()
        scaler.fit(self.train_Xtree)
        self.train_Xtree = scaler.transform(self.train_Xtree)
        self.test_Xtree  = scaler.transform(self.test_Xtree)
        
        # Scale Time features
        self.train_Xtime = self.__normalize_in_time(self.train_Xtime)
        self.test_Xtime  = self.__normalize_in_time(self.test_Xtime)
        
    def __normalize_in_time(self, X: np.array) -> np.array:
        """
        Normalize each sample in time so that the midpoint 0.5 remains fixed, 
        stretching values below toward 0 and above toward 1.
        """
        assert np.issubdtype(X.dtype, np.floating), "Input X must be a float array"
        
        center = 0.5

        # per‐sample min/max along time
        mins = X.min(axis=1, keepdims=True)
        maxs = X.max(axis=1, keepdims=True)

        # distances from center
        den_above = maxs - center
        den_below = center - mins

        # avoid div-by-zero
        eps = 1e-6
        den_above = np.where(np.abs(den_above)<eps, 1.0, den_above)
        den_below = np.where(np.abs(den_below)<eps, 1.0, den_below)

        # piecewise linear stretch around center
        above = X > center
        return np.where(
            above,
            center + (X - center)/den_above * (1.0 - center),
            center - (center - X)/den_below * (      center)
        )
    
    def __add_target(self, meta_pl_all: pl.DataFrame, days_After: int):
        meta_pl_all = (meta_pl_all
            .with_row_index("row_index")
            .with_columns((pl.col("date") + pl.duration(days=days_After)).alias("target_date"))
            .sort(["ticker","date"])
        )
        prices = (meta_pl_all
            .select(["ticker","date","Close"])
            .rename({"date":"price_date","Close":"target_close"})
            .sort(["ticker","price_date"])
        )
        meta_pl_all = (
            meta_pl_all.join_asof(
                prices,
                left_on="target_date",
                right_on="price_date",
                by="ticker",
                strategy="forward",
                check_sortedness=False,
            )
        )
        
        meta_pl_all = meta_pl_all.sort("row_index").drop(["price_date", "row_index"])
        
        return meta_pl_all
            
    def __forge_filtermask(self, feats_tree: np.array, names_tree: np.array) -> np.array:
        mask = np.zeros(feats_tree.shape[0], dtype=bool)
        
        if not self.params["TreeTime_isFiltered"]:
            return np.ones(feats_tree.shape[0], dtype=bool)
        
        if self.params["TreeTime_FourierRSME_q"] is not None:
            idx = np.where(names_tree == 'Fourier_Price_RSMERatioCoeff_1_MH_2')[0]
            arr = feats_tree[:, idx].flatten()
            quant = np.quantile(arr, self.params["TreeTime_FourierRSME_q"])
            mask = mask | (arr <= quant)
            
        if self.params["TreeTime_trend_stc_q"] is not None:
            idx = np.where(names_tree == 'FeatureTA_trend_stc')[0]
            arr = feats_tree[:, idx].flatten()
            quant_lower = np.quantile(arr, self.params["TreeTime_trend_stc_q"])
            mask = mask | (arr <= quant_lower)
            
        if self.params["TreeTime_trend_mass_index_q"] is not None:
            idx = np.where(names_tree == 'FeatureTA_trend_mass_index')[0]
            arr = feats_tree[:, idx].flatten()
            quant_lower = np.quantile(arr, self.params["TreeTime_trend_mass_index_q"])
            mask = mask | (arr <= quant_lower)
            
        if self.params["TreeTime_AvgReturnPct_qup"] is not None:
            idx = np.where(names_tree == 'FeatureGroup_AvgReturnPct')[0]
            arr = feats_tree[:, idx].flatten()
            quant = np.quantile(arr, self.params["TreeTime_AvgReturnPct_qup"])
            mask = mask | (arr >= quant)
            
        if self.params["TreeTime_volatility_atr_qup"] is not None:
            idx = np.where(names_tree == 'FeatureTA_volatility_atr')[0]
            arr = feats_tree[:, idx].flatten()
            quant = np.quantile(arr, self.params["TreeTime_volatility_atr_qup"])
            mask = mask | (arr >= quant)
            
        if self.params["TreeTime_ReturnLog_RSMECoeff_2_qup"] is not None:
            lag = int(self.daysAfter * (5/7))
            candidates = [
                    f'Fourier_ReturnLog_RSMECoeff_2_MH_{m}_lag_m{lag}' for m in [4,5,6,7,8]
                ] + [
                    f'Fourier_ReturnLog_RSMECoeff_2_MH_{m}_lag_m{lag+1}' for m in [4,5,6,7,8]
                ]
            name = next((c for c in candidates if c in names_tree), None)
            if name is not None:
                idx = np.where(names_tree == name)[0]
                arr = feats_tree[:, idx].flatten()
                quant = np.quantile(arr, self.params["TreeTime_ReturnLog_RSMECoeff_2_qup"])
                mask = mask | (arr >= quant)
            else:
                logger.warning("No Fourier ReturnLog RSME Coeff feature found in the dataset. Skipping filtering by Fourier ReturnLog RSME Coeff.")
            
        if self.params["TreeTime_Drawdown_q"] is not None:
            candidates = [
                'MathFeature_Drawdown_MH4',
                'MathFeature_Drawdown_MH5',
                'MathFeature_Drawdown_MH6',
            ]
            name = next((c for c in candidates if c in names_tree), None)
            if name is not None:
                idx = np.where(names_tree == name)[0]
                arr = feats_tree[:, idx].flatten()
                quant_lower = np.quantile(arr, self.params["TreeTime_Drawdown_q"])
                mask = mask | (arr <= quant_lower)
            else:
                logger.warning("No Drawdown feature found in the dataset. Skipping filtering by Drawdown.")
            
        if mask.sum() == 0:
            raise ValueError("No features were selected by filtering.")
        
        return mask
        
    
    def pipeline(self, params: dict = {}, lstm_model = None, lgb_model = None) -> dict:
        """
        Common pipeline steps shared by both analyze() and predict().
        Returns a dictionary of all relevant masked data, trained model, and predictions.
        """
        self.params = {**self.params, **(params or {})}
        if self.train_Xtime is None:
            raise ValueError("Data is not prepared. Please run prepareData() first.")

        # Log distributions
        logger.info(f"Number of time features: {len(self.featureTimeNames)}")
        logger.info("Overall Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.train_ytime > 0.51)

        # LSTM model
        if self.params['TreeTime_run_lstm']:
            if lstm_model is None:
                startTime  = datetime.datetime.now()
                lstm_model = self.__run_LSTM()
                logger.info(f"LSTM completed in {datetime.datetime.now() - startTime}.")

            # LSTM Predictions
            startTime  = datetime.datetime.now()
            y_train_pred = lstm_model.predict(self.train_Xtime, batch_size=self.params['TreeTime_lstm_batch_size'])[:,0]
            y_test_pred = lstm_model.predict(self.test_Xtime, batch_size=self.params['TreeTime_lstm_batch_size'])[:,0]
            rsme = np.sqrt(np.mean((y_train_pred - self.train_ytime) ** 2))
            logger.info(f"  Train (lstm_pred-ytime)  -> RSME: {rsme:.4f}")
            #epsi = 1e-5
            #y_train_pred = np.arctanh(np.clip((y_train_pred - 0.5) * 2.0, -1.0+epsi, 1.0-epsi)) + 1.0
            #y_test_pred  = np.arctanh(np.clip((y_test_pred  - 0.5) * 2.0, -1.0+epsi, 1.0-epsi)) + 1.0
            y_train_pred = (y_train_pred-0.5)/5.0+1.0
            y_test_pred  = (y_test_pred-0.5)/5.0+1.0
            rsme = np.sqrt(np.mean((y_train_pred - self.train_ytree) ** 2))
            logger.info(f"  Train (arctanh-ytree)    -> RSME: {rsme:.4f}")
            logger.info(f"LSTM Prediction completed in {datetime.datetime.now() - startTime}.")
            
            ## Add LSTM predictions to the tree test set
            self.test_Xtree = np.hstack((self.test_Xtree, y_test_pred.reshape(-1, 1)))
            self.train_Xtree = np.hstack((self.train_Xtree, y_train_pred.reshape(-1, 1)))
            self.featureTreeNames = np.hstack((self.featureTreeNames,["LSTM_Prediction"]))
        
        ## Establish weights for Tree
        if self.params['TreeTime_MatchFeatures_run'] is False:
            self.tree_weights = np.ones(self.train_Xtree.shape[0])
        else:
            startTime  = datetime.datetime.now()
            self.tree_weights = self.__establish_weights()
            logger.info(f"Weights established in {datetime.datetime.now() - startTime}.")
        
        # LGB model
        if lgb_model is None:
            startTime  = datetime.datetime.now()
            lgb_model = self.__run_LGB()
            logger.info(f"LGB completed in {datetime.datetime.now() - startTime}.")
        
        # LGB Predictions
        startTime  = datetime.datetime.now()
        y_test_pred = lgb_model.predict(self.test_Xtree, num_iteration=lgb_model.best_iteration)
        y_train_pred = lgb_model.predict(self.train_Xtree, num_iteration=lgb_model.best_iteration)
        rsme = np.sqrt(np.mean((y_train_pred - self.train_ytree) ** 2))
        logger.info(f"  Train (lgbm_pred - ytree)       -> RSME: {rsme:.4f}")
        logger.info(f"LGB Prediction completed in {datetime.datetime.now() - startTime}.")
        
        # filter unimportant features
        # to be implemented

        # Return everything needed
        return {
            'lstm_model': lstm_model,
            'lgb_model': lgb_model,
            'y_test_pred': y_test_pred,
            'y_train_pred': y_train_pred,
        }

    def __run_LSTM(self):
        # Hyperparameters to tune
        lstm_units = self.params["TreeTime_lstm_units"]
        num_layers = self.params["TreeTime_lstm_num_layers"]
        dropout = self.params["TreeTime_lstm_dropout"]
        recurrent_dropout = self.params["TreeTime_lstm_recurrent_dropout"]
        learning_rate = self.params["TreeTime_lstm_learning_rate"]
        optimizer_name = self.params["TreeTime_lstm_optimizer"]
        bidirectional = self.params["TreeTime_lstm_bidirectional"]
        batch_size = self.params["TreeTime_lstm_batch_size"]
        epochs = self.params["TreeTime_lstm_epochs"]
        loss_name = self.params["TreeTime_lstm_loss"]
        
        # Regularization hyperparameters
        l1 = self.params.get("TreeTime_lstm_l1", 0.0)
        l2 = self.params.get("TreeTime_lstm_l2", 0.0)
        inter_dropout = self.params.get("TreeTime_inter_dropout", 0.0)
        noise_std = self.params.get("TreeTime_input_gaussian_noise", 0.0)
        
        # Conv1D option
        use_conv1d = self.params.get("TreeTime_lstm_conv1d", False)
        conv_filters = lstm_units
        conv_kernel = self.params.get("TreeTime_lstm_conv1d_kernel_size", 3)

        X_full, y_full = self.train_Xtime, self.train_ytime
        n_total = X_full.shape[0]
        split_at = int(n_total * 0.95)
        X_train, X_holdout = X_full[:split_at], X_full[split_at:]
        y_train, y_holdout = y_full[:split_at], y_full[split_at:]
        
        # Build model
        model = Sequential([Input(shape=self.train_Xtime.shape[1:])])
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
            validation_data=(X_holdout, y_holdout),
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

        return model    
    
    def __run_LGB(self):
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
#
        #gbm.best_iteration = best_it
        #gbm.best_score['test']['ndcg@5'] = best_score
        return gbm
        
    def __establish_weights(self) -> np.array:
        nSamples = self.train_Xtree.shape[0]
        mask_sparsing = np.random.rand(nSamples) <= 1e4/np.sum(nSamples) # for speedup
        
        ksDist = DistributionTools.ksDistance(
            self.train_Xtree[mask_sparsing].astype(np.float64), 
            self.test_Xtree.copy().astype(np.float64), 
            weights=None,
            overwrite=True)
        
        logger.info(f"  Train-Test Distri Equality: Mean: {np.mean(ksDist)}, Quantile 0.9: {np.quantile(ksDist, 0.9)}")

        indices_masked = self.__establish_matching_featureindices(ksDist)
        tree_weights = DistributionTools.establishMatchingWeight(
            self.train_Xtree[:, indices_masked].astype(np.float64),
            self.test_Xtree[:, indices_masked].astype(np.float64),
            n_bin = 15,
            minbd = self.params['TreeTime_MatchFeatures_minWeight']
        )
        tree_weights *= (self.train_Xtree.shape[0] / np.sum(tree_weights))
        
        logger.debug(f"  Zeros Weight Ratio: {np.sum(tree_weights < 1e-6) / len(tree_weights)}")
        logger.debug(f"  Negative Weight Ratio: {np.sum(tree_weights < -1e-5) / len(tree_weights)}")
        logger.debug(f"  Mean Weight: {np.mean(tree_weights)}")
        logger.debug(f"  Quantile 0.1 Weight: {np.quantile(tree_weights, 0.1)}")
        logger.debug(f"  Quantile 0.9 Weight: {np.quantile(tree_weights, 0.9)}")
        
        return tree_weights

    def __establish_matching_featureindices(self, ksDist) -> np.array:
        assert ksDist.shape[0] == self.featureTreeNames.shape[0], "Mismatch in matching feature function."
        
        nFeat = self.featureTreeNames.shape[0]
        mask_colToMatch = np.zeros(nFeat, dtype=bool)
        
        if self.params["TreeTime_MatchFeatures_Pricediff"]:
            mask_colToMatch |= np.char.find(self.featureTreeNames, "MathFeature_Price_Diff") >= 0
            
        if self.params["TreeTime_MatchFeatures_FinData_quar"]:
            mask_colToMatch |= np.char.find(self.featureTreeNames, "FinData_quar") >= 0
            
        if self.params["TreeTime_MatchFeatures_FinData_metrics"]:
            mask_colToMatch |= np.char.find(self.featureTreeNames, "FinData_metrics") >= 0
        
        if self.params["TreeTime_MatchFeatures_Fourier_RSME"]:
            mask_colToMatch |= np.char.find(self.featureTreeNames, "Fourier_Price_RSME") >= 0
            
        if self.params["TreeTime_MatchFeatures_Fourier_Sign"]:
            mask_colToMatch |= np.char.find(self.featureTreeNames, "Fourier_Price_Sign") >= 0
            
        if self.params["TreeTime_MatchFeatures_TA_trend"]:
            mask_colToMatch |= np.char.find(self.featureTreeNames, "FeatureTA_trend") >= 0
            
        if self.params["TreeTime_MatchFeatures_FeatureGroup_VolGrLvl"]:
            mask_colToMatch |= np.char.find(self.featureTreeNames, "FeatureGroup_VolGrLvl") >= 0
            
        if self.params["TreeTime_MatchFeatures_LSTM_Prediction"]:
            mask_colToMatch |= np.char.find(self.featureTreeNames, "LSTM_Prediction") >= 0
            
        if all(~mask_colToMatch):
            mask_colToMatch = np.char.find(self.featureTreeNames, "MathFeature_Price_Diff") >= 0
        
        idces = np.arange(mask_colToMatch.shape[0])[mask_colToMatch]
        idces = idces[np.argsort(ksDist[mask_colToMatch])]
        top_idces = idces[-self.params['TreeTime_MatchFeatures_truncation']:]
            
        return top_idces
    
    def analyze(self, params: dict = {}, lstm_model = None, lgb_model = None, logger_disabled: bool = False) -> tuple:
        # Run common pipeline in "analyze" mode
        data = self.pipeline(params = params, lstm_model = lstm_model, lgb_model = lgb_model)

        logger.disabled = logger_disabled
        
        if data["lgb_model"] is not None:
            ModelAnalyzer.print_feature_importance_LGBM(data['lgb_model'], self.featureTreeNames, 15)
        
        # Additional analysis with test set
        y_test_pred: np.array = data['y_test_pred']
        y_train_pred: np.array = data['y_train_pred']
        
        mean_topval_per_Day = np.zeros(len(self.test_dates))
        mean_toppred_per_Day = np.zeros(len(self.test_dates))
        for idx, test_date in enumerate(self.test_dates):
            logger.info(f"Analyzing test date: {test_date}")
            meta_pl_test_ondate = self.meta_pl_test.filter(pl.col("date") == test_date)
            if meta_pl_test_ondate.is_empty():
                logger.error(f"ERROR: No data available for test date {test_date}. Skipping analysis.")
                continue
            
            mask_date = (self.meta_pl_test["date"] == test_date).to_numpy()
            y_test_pred_ondate = y_test_pred[mask_date]
            
            # Log test label distribution & classification metrics
            #logger.info("Testing Masked Classification Metrics:")
            #ModelAnalyzer.print_classification_metrics(
            #    self.test_ytree[mask_date] > 1.02,
            #    y_test_pred_ondate > 1.02,
            #    None
            #)

            logger.info("Reporting Error on all filtered:")
            error = (self.test_ytree[mask_date] - y_test_pred_ondate)
            logger.info(f"  Mean Root Squared Error: {np.sqrt(np.mean((error) ** 2)):.4f}")
            logger.info(f"  Median Absolute Error: {np.median(np.abs(error)):.4f}")
            logger.info(f"  Median Error: {np.mean(error):.4f}")
            logger.info(f"  0.9 Quantile Error: {np.quantile(error, 0.9):.4f}")
            logger.info(f"  0.1 Quantile Error: {np.quantile(error, 0.1):.4f}")

            # Top m analysis
            logger.info("Reporting Error on top predicted:")
            m = self.params['TreeTime_top_highest']
            top_m_indices = np.flip(np.argsort(y_test_pred_ondate)[-m:])
            selected_df = meta_pl_test_ondate.with_columns(pl.Series("prediction_ratio", y_test_pred_ondate))[top_m_indices]
            
            selected_true_values_reg = selected_df["target_ratio"].to_numpy().flatten()
            selected_pred_values_reg = y_test_pred_ondate[top_m_indices]
            logger.info(f"  Mean value of top {m}: {np.mean(selected_true_values_reg)}")
            logger.info(f"  Min value of top {m}: {np.min(selected_true_values_reg)}")
            logger.info(f"  Max value of top {m}: {np.max(selected_true_values_reg)}")
            logger.info(f"  Mean pred value of top {m}: {np.mean(selected_pred_values_reg)}")
        
            with pl.Config(ascii_tables=True):
                logger.info(f"DataFrame:\n{selected_df}")
                
            mean_topval_per_Day[idx] = np.mean(selected_true_values_reg)
            mean_toppred_per_Day[idx] = np.mean(selected_pred_values_reg)
        
        inliner_mask = (~np.isnan(mean_toppred_per_Day)) & (~np.isnan(mean_topval_per_Day))
        inliner_mask = inliner_mask & (mean_topval_per_Day > 0.5)
        mean_topval_per_Day = mean_topval_per_Day[inliner_mask]
        mean_toppred_per_Day = mean_toppred_per_Day[inliner_mask]
        try:
            act_arr = mean_topval_per_Day
            pred_arr = mean_toppred_per_Day
            _, _, r_value, _, _ = stats.linregress(
                (pred_arr - np.mean(pred_arr))/np.sqrt(np.var(pred_arr)), 
                (act_arr - np.mean(act_arr))/np.sqrt(np.var(act_arr))
            )
            r2_score = r_value**2
        except Exception as e:
            logger.error(f"Error calculating R² score: {e}")
            r2_score = np.nan
            logger.error("Error calculating R² score. Returning NaN.")
        
        # Print all values

        n_arr = len(mean_topval_per_Day)
        logger.info("Mean Top Value per Day ALL VALUES: [" + ", ".join([f"{v:.4f}" for v in mean_topval_per_Day]) + "]")
        logger.info(f"Mean Top Predicted per Day ALL VALUES: [" + ", ".join([f"{v:.4f}" for v in mean_toppred_per_Day])+ "]")
        #logger.info(f"quant3(pred): {np.quantile(mean_toppred_per_Day, 0.3):.4f}")
        #logger.info(f"quant5(pred): {np.quantile(mean_toppred_per_Day, 0.5):.4f}")
        #logger.info(f"quant7(pred): {np.quantile(mean_toppred_per_Day, 0.7):.4f}")
        #logger.info(f"quant9(pred): {np.quantile(mean_toppred_per_Day, 0.9):.4f}")
        #pred_half_3 = np.quantile(mean_toppred_per_Day[n_arr//2:], 0.3)
        #pred_half_5 = np.quantile(mean_toppred_per_Day[n_arr//2:], 0.5)
        #pred_half_7 = np.quantile(mean_toppred_per_Day[n_arr//2:], 0.7)
        #pred_half_9 = np.quantile(mean_toppred_per_Day[n_arr//2:], 0.9)
        #logger.info(f"quant3(pred half): {pred_half_3:.4f}")
        #logger.info(f"quant5(pred half): {pred_half_5:.4f}")
        #logger.info(f"quant7(pred half): {pred_half_7:.4f}")
        #logger.info(f"quant9(pred half): {pred_half_9:.4f}")
        #logger.info(f"quant3(train pred): {np.quantile(y_train_pred, 0.3):.4f}")
        #logger.info(f"quant5(train pred): {np.quantile(y_train_pred, 0.5):.4f}")
        #logger.info(f"quant7(train pred): {np.quantile(y_train_pred, 0.7):.4f}")
        #logger.info(f"quant9(train pred): {np.quantile(y_train_pred, 0.9):.4f}")
        #logger.info(f"Mean actual value for pred > quant3(pred): {np.mean(mean_topval_per_Day[:n_arr//2][mean_toppred_per_Day[:n_arr//2] > pred_half_3]):.4f}")
        #logger.info(f"Mean actual value for pred > quant5(pred): {np.mean(mean_topval_per_Day[:n_arr//2][mean_toppred_per_Day[:n_arr//2] > pred_half_5]):.4f}")
        #logger.info(f"Mean actual value for pred > quant7(pred): {np.mean(mean_topval_per_Day[:n_arr//2][mean_toppred_per_Day[:n_arr//2] > pred_half_7]):.4f}")
        #logger.info(f"Mean actual value for pred > quant9(pred): {np.mean(mean_topval_per_Day[:n_arr//2][mean_toppred_per_Day[:n_arr//2] > pred_half_9]):.4f}")
        
        logger.info(f"Best Train score: {data['lgb_model'].best_score['train']}")
        logger.info(f"Best Test score: {data['lgb_model'].best_score['test']}")
        logger.info(f"Best Iteration: {data['lgb_model'].best_iteration}")
        return (
            np.mean(mean_topval_per_Day), 
            np.mean(mean_toppred_per_Day),
        )
        
    def predict(self, params: dict = {}, lstm_model = None, lgb_model = None) -> tuple:
        # Run common pipeline in "analyze" mode
        data = self.pipeline(params = params, lstm_model = lstm_model, lgb_model = lgb_model)

        if data["lgb_model"] is not None:
            ModelAnalyzer.print_feature_importance_LGBM(data['lgb_model'], self.featureTreeNames, 15)
        
        # Additional analysis with test set
        y_test_pred: np.array = data['y_test_pred']
        y_train_pred: np.array = data['y_train_pred']
        
        mean_topval_per_Day = np.zeros(len(self.test_dates))
        mean_toppred_per_Day = np.zeros(len(self.test_dates))
        for idx, test_date in enumerate(self.test_dates):
            logger.info(f"Analyzing test date: {test_date}")
            meta_pl_test_ondate = self.meta_pl_test.filter(pl.col("date") == test_date)
            if meta_pl_test_ondate.is_empty():
                logger.error(f"ERROR: No data available for test date {test_date}. Skipping analysis.")
                continue
            
            mask_date = (self.meta_pl_test["date"] == test_date).to_numpy()
            y_test_pred_ondate = y_test_pred[mask_date]

            logger.info("Reporting Error on all filtered:")
            error = (self.test_ytree[mask_date] - y_test_pred_ondate)
            logger.info(f"  Mean Root Squared Error: {np.sqrt(np.mean((error) ** 2)):.4f}")
            logger.info(f"  Median Absolute Error: {np.median(np.abs(error)):.4f}")
            logger.info(f"  Median Error: {np.mean(error):.4f}")
            logger.info(f"  0.9 Quantile Error: {np.quantile(error, 0.9):.4f}")
            logger.info(f"  0.1 Quantile Error: {np.quantile(error, 0.1):.4f}")

            # Top m analysis
            logger.info("Reporting Error on top predicted:")
            m = self.params['TreeTime_top_highest']
            top_m_indices = np.flip(np.argsort(y_test_pred_ondate)[-m:])
            selected_df = meta_pl_test_ondate.with_columns(pl.Series("prediction_ratio", y_test_pred_ondate))[top_m_indices]
            
            selected_true_values_reg = selected_df["target_ratio"].to_numpy().flatten()
            selected_pred_values_reg = y_test_pred_ondate[top_m_indices]
            logger.info(f"  Mean value of top {m}: {np.mean(selected_true_values_reg)}")
            logger.info(f"  Min value of top {m}: {np.min(selected_true_values_reg)}")
            logger.info(f"  Max value of top {m}: {np.max(selected_true_values_reg)}")
            logger.info(f"  Mean pred value of top {m}: {np.mean(selected_pred_values_reg)}")
        
            with pl.Config(ascii_tables=True):
                logger.info(f"DataFrame:\n{selected_df}")
                
            mean_topval_per_Day[idx] = np.mean(selected_true_values_reg)
            mean_toppred_per_Day[idx] = np.mean(selected_pred_values_reg)
        
        inliner_mask = (~np.isnan(mean_toppred_per_Day)) & (~np.isnan(mean_topval_per_Day))
        inliner_mask = inliner_mask & (mean_topval_per_Day > 0.5)
        mean_topval_per_Day = mean_topval_per_Day[inliner_mask]
        mean_toppred_per_Day = mean_toppred_per_Day[inliner_mask]
        
        logger.info("Mean Top Value per Day ALL VALUES: [" + ", ".join([f"{v:.4f}" for v in mean_topval_per_Day]) + "]")
        logger.info(f"Mean Top Predicted per Day ALL VALUES: [" + ", ".join([f"{v:.4f}" for v in mean_toppred_per_Day])+ "]")

        logger.info(f"Best Train score: {data['lgb_model'].best_score['train']}")
        logger.info(f"Best Test score: {data['lgb_model'].best_score['test']}")
        logger.info(f"Best Iteration: {data['lgb_model'].best_iteration}")
        return (
            np.mean(mean_topval_per_Day), 
            np.mean(mean_toppred_per_Day), 
        )