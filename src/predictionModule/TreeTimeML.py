import os
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import logging
import datetime
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GaussianNoise, LSTM, Bidirectional, Dropout, Dense, Conv1D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError

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
            test_date: datetime.date,
            group: str,
            params: dict = None,
        ):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.group = group
        self.train_start_date = train_start_date
        self.test_date = test_date
        
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
        years = np.arange(self.train_start_date.year, self.test_date.year + 1)
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
            
        testdate_in_db = False
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
            
            if self.test_date in meta_tree["date"]:
                testdate_in_db = True
                
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
        
        if not testdate_in_db:
            logger.warning(f"Test date {self.test_date} not found in the database. Resetting to last trading day.")
            self.test_date = meta_pl_all.filter(pl.col("date") <= self.test_date).select("date").max()["date"].item()
        
        # get target value
        meta_pl_all = self.__add_target(meta_pl_all, self.daysAfter)
        meta_pl_all = meta_pl_all.with_columns((pl.col("target_close")  / pl.col("Close")).alias("target_ratio"))
        tar_all = meta_pl_all["target_close"].to_numpy().flatten()
        cur_all = meta_pl_all["Close"].to_numpy().flatten()
        tar_masked = tar_all[mask_all]
        cur_masked = cur_all[mask_all]
        meta_pl = meta_pl_all.filter(mask_all)
        
        # Filter start date end date
        mask_at_test_date = (meta_pl["date"] == self.test_date).to_numpy()
        mask_inbetween_date = ((meta_pl["date"] >= self.train_start_date) & (meta_pl["date"] <= self.test_date - pd.Timedelta(days=self.daysAfter))).to_numpy()
        
        rat_at_test_date = tar_masked[mask_at_test_date] / cur_masked[mask_at_test_date]
        rat_inbetween = tar_masked[mask_inbetween_date] / cur_masked[mask_inbetween_date]
        
        self.meta_pl_train = meta_pl.filter(mask_inbetween_date)
        self.meta_pl_test  = meta_pl.filter(mask_at_test_date)
        
        self.test_Xtree = all_Xtree_pre[mask_at_test_date]
        self.test_Xtime = all_Xtime_pre[mask_at_test_date]
        
        self.test_ytree = rat_at_test_date
        self.test_ytime = np.tanh(rat_at_test_date-1.0)/2.0 + 0.5
        
        self.train_Xtree = all_Xtree_pre[mask_inbetween_date]
        self.train_Xtime = all_Xtime_pre[mask_inbetween_date]
        
        self.train_ytree = rat_inbetween
        self.train_ytime = np.tanh(rat_inbetween-1.0)/2.0 + 0.5
        
        # Scale Tree features
        scaler = StandardScaler()
        scaler.fit(self.train_Xtree)
        self.train_Xtree = scaler.transform(self.train_Xtree)
        self.test_Xtree  = scaler.transform(self.test_Xtree)
    
    def __add_target(self, meta_pl_all: pl.DataFrame, days_After: int):
        meta_pl_all = (meta_pl_all
            .with_row_index("row_index")
            .with_columns((pl.col("date") + pl.duration(days=days_After)).alias("target_date"))
            .sort(["ticker","date"])
        )
        prices = meta_pl_all.select(["ticker","date","Close"]).rename({"date":"price_date","Close":"target_close"}).sort(["ticker","price_date"])
        meta_pl_all = (
            meta_pl_all.join_asof(
                prices,
                left_on="target_date",
                right_on="price_date",
                by="ticker",
                strategy="backward",
                check_sortedness=False,
            )
        )
        
        meta_pl_all = meta_pl_all.sort("row_index").drop(["price_date", "row_index"])
        
        return meta_pl_all
            
    def __forge_filtermask(self, feats_tree: np.array, names_tree: np.array) -> np.array:
        mask = np.zeros(feats_tree.shape[0], dtype=bool)
        
        if not self.params["TreeTime_isFiltered"]:
            return np.ones(feats_tree.shape[0], dtype=bool)
        
        if not self.params["TreeTime_FourierRSME_q"] is None:
            idx = np.where(names_tree == 'Fourier_Price_RSMERatioCoeff_1_MH_1')[0]
            idx1 = np.where(names_tree == 'FeatureTA_Open_lag_m1')[0]
            idx2 = np.where(names_tree == 'FeatureTA_Open_lag_m10')[0]
            arr = feats_tree[:, idx].flatten()
            arr1 = feats_tree[:, idx1].flatten()
            arr2 = feats_tree[:, idx2].flatten()
            mask = mask | (arr2 < arr1*0.98)
            quant = np.quantile(arr[(arr2 < arr1*0.98)], self.params["TreeTime_FourierRSME_q"])
            mask = mask | (arr <= quant)
            
        if not self.params["TreeTime_RSIExt_q"] is None:
            idx = np.where(names_tree == 'FeatureTA_momentum_stoch_rsi')[0]
            arr = feats_tree[:, idx].flatten()
            quant_lower = np.quantile(arr, self.params["TreeTime_RSIExt_q"])
            quant_upper = np.quantile(arr, 1-self.params["TreeTime_RSIExt_q"])
            mask = mask | (arr <= quant_lower) | (arr >= quant_upper)
            
        if mask.sum() == 0:
            raise ValueError("No features were selected by filtering.")
        
        return mask
        
    
    def pipeline(self):
        """
        Common pipeline steps shared by both analyze() and predict().
        Returns a dictionary of all relevant masked data, trained model, and predictions.
        """
        if self.train_Xtime is None:
            raise ValueError("Data is not prepared. Please run prepareData() first.")

        # Log distributions
        logger.info(f"Number of time features: {len(self.featureTimeNames)}")
        logger.info("Overall Training Label Distribution:")
        ModelAnalyzer().print_label_distribution(self.train_ytime > 0.51)

        # LSTM model
        startTime  = datetime.datetime.now()
        lstm_model = self.__run_LSTM()
        logger.info(f"LSTM completed in {datetime.datetime.now() - startTime}.")

        # LSTM Predictions
        startTime  = datetime.datetime.now()
        y_train_pred = lstm_model.predict(self.train_Xtime, batch_size=self.params['TreeTime_lstm_batch_size'])[:,0]
        y_test_pred = lstm_model.predict(self.test_Xtime, batch_size=self.params['TreeTime_lstm_batch_size'])[:,0]
        rsme = np.sqrt(np.mean((y_train_pred - self.train_ytime) ** 2))
        logger.info(f"  Train (lstm_pred-ytime)  -> RSME: {rsme:.4f}")
        y_train_pred = np.arctanh((y_train_pred - 0.5) * 2.0) + 1.0
        y_test_pred  = np.arctanh((y_test_pred  - 0.5) * 2.0) + 1.0
        rsme = np.sqrt(np.mean((y_train_pred - self.train_ytree) ** 2))
        logger.info(f"  Train (arctanh-ytree)    -> RSME: {rsme:.4f}")
        logger.info(f"LSTM Prediction completed in {datetime.datetime.now() - startTime}.")
        
        ## Add LSTM predictions to the tree test set
        self.test_Xtree = np.hstack((self.test_Xtree, y_test_pred.reshape(-1, 1)))
        self.train_Xtree = np.hstack((self.train_Xtree, y_train_pred.reshape(-1, 1)))
        self.featureTreeNames = np.hstack((self.featureTreeNames,["LSTM_Prediction"]))
        
        ## Establish weights for Tree
        startTime  = datetime.datetime.now()
        self.tree_weights = self.__establish_weights()
        logger.info(f"Weights established in {datetime.datetime.now() - startTime}.")
        
        # LGB model
        startTime  = datetime.datetime.now()
        lgb_model = self.__run_LGB()
        logger.info(f"LGB completed in {datetime.datetime.now() - startTime}.")
        
        # LGB Predictions
        startTime  = datetime.datetime.now()
        y_train_pred = lgb_model.predict(self.train_Xtree, num_iteration=lgb_model.best_iteration)
        y_test_pred = lgb_model.predict(self.test_Xtree, num_iteration=lgb_model.best_iteration)
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
        model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)))
        
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
        if loss_name == "quantile":
            loss = quantile_loss(0.9)
        elif loss_name == "mse":
            loss = MeanSquaredError()
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[MeanSquaredError(name='mse'),
                    RootMeanSquaredError(name='rmse')]
        )
        # Callbacks
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_holdout, y_holdout),
            callbacks=[es, rlrop],
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
            'objective': 'regression',
            #'alpha': 0.85,
            'metric': 'l2_root',  # NOTE: the string 'rsme' is not recognized, v 4.5.0
            'lambda_l1': self.params['TreeTime_lgb_lambda_l1'],
            'lambda_l2': self.params['TreeTime_lgb_lambda_l2'],
            'early_stopping_rounds': num_boost_round//10,
            'feature_fraction': self.params['TreeTime_lgb_feature_fraction'],
            'num_leaves': self.params['TreeTime_lgb_num_leaves'], 
            'max_depth': self.params['TreeTime_lgb_max_depth'],
            'learning_rate': self.params['TreeTime_lgb_learning_rate'],
            'min_data_in_leaf': self.params['TreeTime_lgb_min_data_in_leaf'],
            'min_gain_to_split': self.params['TreeTime_lgb_min_gain_to_split'],
            'path_smooth': self.params['TreeTime_lgb_path_smooth'],
            'min_sum_hessian_in_leaf': self.params['TreeTime_lgb_min_sum_hessian_in_leaf'],
            'random_state': 41,
        }   
        
        n_total = self.train_Xtree.shape[0]
        split_at = int(n_total * 0.95)
        mask_val = np.zeros(n_total, dtype=bool)
        mask_val[split_at:] = True

        train_data = lgb.Dataset(self.train_Xtree[~mask_val], label = self.train_ytree[~mask_val], weight=self.tree_weights[~mask_val])
        test_data = lgb.Dataset(self.train_Xtree[mask_val], label = self.train_ytree[mask_val], reference=train_data)
        
        def print_eval_after_100(env):
            if env.iteration % 100 == 0 or env.iteration == num_boost_round:
                results = [
                    f"{data_name}'s {eval_name}: {result}"
                    for data_name, eval_name, result, _ in env.evaluation_result_list
                ]
                logger.info(f"Iteration {env.iteration}: " + ", ".join(results))
        
        gbm = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=num_boost_round,
            callbacks=[print_eval_after_100]
        )
        
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
        
        idces = np.arange(mask_colToMatch.shape[0])[mask_colToMatch]
        idces = idces[np.argsort(ksDist[mask_colToMatch])]
        top_idces = idces[-self.params['TreeTime_MatchFeatures_truncation']:]
        
        if len(top_idces) == 0:
            top_idces = self.featureTreeNames.shape[0] -1
            
        return top_idces
    
    def analyze(self, data = None):
        # Run common pipeline in "analyze" mode
        if data is None:
            # If no data is provided, run the pipeline
            data = self.pipeline()

        # Additional analysis with test set
        y_test_pred: np.array = data['y_test_pred']

        # Log test label distribution & classification metrics
        logger.info("Testing Masked Classification Metrics:")
        ModelAnalyzer().print_classification_metrics(
            self.test_ytree > 1.02,
            y_test_pred > 1.02,
            None
        )

        logger.info("Testing Errors:")
        error = (self.test_ytree - y_test_pred)
        logger.info(f"Mean Root Squared Error: {np.sqrt(np.mean((error) ** 2)):.4f}")
        logger.info(f"Median Absolute Error: {np.median(np.abs(error)):.4f}")
        logger.info(f"Median Error: {np.mean(error):.4f}")
        logger.info(f"0.9 Quantile Error: {np.quantile(error, 0.9):.4f}")
        logger.info(f"0.1 Quantile Error: {np.quantile(error, 0.1):.4f}")

        # Top m analysis
        m = self.params['TreeTime_top_highest']
        top_m_indices = np.flip(np.argsort(y_test_pred)[-m:])
        selected_df = self.meta_pl_test.with_columns(pl.Series("prediction_ratio", y_test_pred))[top_m_indices]
        
        selected_true_values_reg = selected_df["target_ratio"].to_numpy().flatten()
        selected_pred_values_reg = y_test_pred[top_m_indices]
        logger.info(f"Mean value of top {m}: {np.mean(selected_true_values_reg)}")
        logger.info(f"Min value of top {m}: {np.min(selected_true_values_reg)}")
        logger.info(f"Max value of top {m}: {np.max(selected_true_values_reg)}")
        logger.info(f"Mean pred value of top {m}: {np.mean(selected_pred_values_reg)}")
        

        with pl.Config(ascii_tables=True):
            logger.info(f"DataFrame:\n{selected_df}")
        
        return (
            np.mean(selected_true_values_reg), 
            np.mean(selected_pred_values_reg),
        )
        
    def predict(self):
        # Run common pipeline in "analyze" mode
        data = self.pipeline()

        # Additional analysis with test set
        y_test_pred: np.array = data['y_test_pred']

        # Top m analysis
        m = self.params['TreeTime_top_highest']
        top_m_indices = np.flip(np.argsort(y_test_pred)[-m:])
        selected_df = self.meta_pl_test.with_columns(pl.Series("prediction_ratio", y_test_pred))[top_m_indices]
        
        selected_pred_values_reg = y_test_pred[top_m_indices]
        logger.info(f"Mean pred value of top {m}: {np.mean(selected_pred_values_reg)}")

        with pl.Config(ascii_tables=True):
            logger.info(f"DataFrame:\n{selected_df}")
        
        return (
            np.mean(selected_pred_values_reg)
        )