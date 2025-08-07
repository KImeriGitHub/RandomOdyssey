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

from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.mathTools.DistributionTools import DistributionTools
from src.predictionModule.LoadupSamples import LoadupSamples
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels

logger = logging.getLogger(__name__)

class TreeTimeML:
    # Class-level default parameters
    DEFAULT_PARAMS = {
        "daysAfterPrediction": None,
        "idxAfterPrediction": 10,
        'timesteps': 20,
        'target_option': 'last',
        
        "LoadupSamples_time_inc_factor": 10,
        "LoadupSamples_tree_scaling_standard": True,
        "LoadupSamples_time_scaling_stretch": True,
    }

    def __init__(
            self, 
            train_start_date: datetime.date,
            test_dates: list[datetime.date],
            group: str,
            params: dict = None,
            loadup: LoadupSamples = None
        ):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.group = group
        self.train_start_date = train_start_date
        self.test_dates = test_dates
        
        self.min_test_date = min(self.test_dates)
        self.max_test_date = max(self.test_dates)
        
        # Assign parameters to instance variables
        if loadup is None or not isinstance(loadup, LoadupSamples):
            ls = LoadupSamples(
                train_start_date=self.train_start_date,
                test_dates=self.test_dates,
                group=self.group,
                params=self.params,
            )
        else:
            ls = loadup
            if ls.test_dates != self.test_dates:
                raise ValueError("Provided LoadupSamples does not match the test dates.")
            self.test_dates = ls.test_dates
            
        self.idxDaysAfter = ls.idxAfter
        self.timesteps = ls.timesteps
        
        self.featureTreeNames = ls.featureTreeNames
        self.featureTimeNames = ls.featureTimeNames
        self.meta_pl_train = ls.meta_pl_train
        self.meta_pl_test = ls.meta_pl_test
        
        self.train_Xtree = ls.train_Xtree
        self.train_Xtime = ls.train_Xtime
        self.train_ytree = ls.train_ytree
        self.train_ytime = ls.train_ytime
        
        self.test_Xtree = ls.test_Xtree
        self.test_Xtime = ls.test_Xtime
        self.test_ytree = ls.test_ytree
        self.test_ytime = ls.test_ytime
        
    
    def pipeline(self, lstm_model = None, lgb_model = None) -> dict:
        """
        Common pipeline steps shared by both analyze() and predict().
        Returns a dictionary of all relevant masked data, trained model, and predictions.
        """        
        # Filter samples
        samples_dates_train = self.meta_pl_train.select(pl.col("date")).to_series()
        samples_dates_test = self.meta_pl_test.select(pl.col("date")).to_series()
        fs = FilterSamples(
            Xtree_train = self.train_Xtree,
            ytree_train = self.train_ytree,
            treenames = self.featureTreeNames,
            Xtree_test = self.test_Xtree,
            samples_dates_train = samples_dates_train,
            samples_dates_test = samples_dates_test,
            ytree_test = self.test_ytree,
            params = self.params
        )
    
        mask_train, mask_test, s_tr, s_te = fs.run()
        
        mm = MachineModels(self.params)

        # LSTM model
        if self.params['TreeTime_run_lstm']:
            logger.info(f"Number of time features: {len(self.featureTimeNames)}")
            inc_factor = self.params["LoadupSamples_time_inc_factor"]
            if lstm_model is None:
                startTime  = datetime.datetime.now()
                required_features = [
                    "MathFeature_TradedPrice",
                    "FeatureTA_High",
                    "FeatureTA_Low",
                    "FeatureTA_volume_obv",
                    "FeatureTA_volume_vpt"
                ]

                # Validate feature presence and retrieve indices
                missing_features = [f for f in required_features if f not in self.featureTimeNames]
                if missing_features:
                    raise ValueError(f"Missing required features in featureTimeNames: {missing_features}")

                idxs_features = [np.where(self.featureTimeNames == feature)[0][0] for feature in required_features]
                lstm_model, lstm_res = mm.run_LSTM_torch(
                    X_train=self.train_Xtime[:, :, idxs_features],
                    y_train=self.train_ytime,
                    X_test=self.test_Xtime[:, :, idxs_features],
                    y_test=self.test_ytime
                )
                logger.info(f"LSTM RSME: {lstm_res['val_rmse'] * 2.0 / inc_factor:.4f}")
                logger.info(f"LSTM completed in {datetime.datetime.now() - startTime}.")

            # LSTM Predictions
            startTime  = datetime.datetime.now()
            y_train_pred = mm.predict_LSTM_torch(
                lstm_model, self.train_Xtime[:, :, idxs_features], batch_size = self.params['LSTM_batch_size'], device='cpu')
            y_test_pred = mm.predict_LSTM_torch(
                lstm_model, self.test_Xtime[:, :, idxs_features], batch_size = self.params['LSTM_batch_size'], device='cpu')

            rsme = np.sqrt(np.mean((y_train_pred - self.train_ytime) ** 2))
            logger.info(f"  Train (lstm_pred-ytime)  -> RSME: {rsme:.4f}")
            
            y_train_pred = (y_train_pred - 0.5) / inc_factor * 2.0 + 1.0
            y_test_pred  = (y_test_pred - 0.5) / inc_factor * 2.0 + 1.0
            rsme = np.sqrt(np.mean((y_train_pred - self.train_ytree) ** 2))
            logger.info(f"  Train (arctanh-ytree)    -> RSME: {rsme:.4f}")
            logger.info(f"LSTM Prediction completed in {datetime.datetime.now() - startTime}.")
            
            ## Add LSTM predictions to the tree test set
            train_std = np.std(y_train_pred)
            test_std = 1.0 if np.std(y_test_pred) < 1e-6 else np.std(y_test_pred)
            self.train_Xtree = np.hstack((self.train_Xtree, ((y_train_pred-1.0)/train_std).reshape(-1, 1)))
            self.test_Xtree = np.hstack((self.test_Xtree, ((y_test_pred-1.0)/test_std).reshape(-1, 1)))
            self.featureTreeNames = np.hstack((self.featureTreeNames, ["LSTM_Prediction"]))
        
        ## Establish weights for Tree
        startTime  = datetime.datetime.now()
        if self.params['TreeTime_MatchFeatures_run']:
            logger.info("Establishing weights for TreeTime features...")
            self.tree_weights = self.__establish_weights()
            logger.info(f"Weights established in {datetime.datetime.now() - startTime}.")
        else:
            logger.info("Skipping TreeTime feature weights establishment.")
            self.tree_weights = np.ones(self.train_Xtree.shape[0], dtype=np.float64)
        
        # LGB model
        if lgb_model is None:
            startTime  = datetime.datetime.now()
            maskless = self.params.get('TreeTime_lgb_maskless', True)
            lgb_model, lgb_res_dict = mm.run_LGB(
                X_train=self.train_Xtree[mask_train] if not maskless else self.train_Xtree,
                y_train=self.train_ytree[mask_train] if not maskless else self.train_ytree,
                X_test=self.test_Xtree[mask_test] if not maskless else self.test_Xtree,
                y_test=self.test_ytree[mask_test] if not maskless else self.test_ytree,
                weights=self.tree_weights[mask_train] if not maskless else self.tree_weights,
            )
            logger.info(f"LGB completed in {datetime.datetime.now() - startTime}.")
        
        # LGB Predictions
        startTime  = datetime.datetime.now()
        y_train_pred_masked = lgb_model.predict(self.train_Xtree[mask_train], num_iteration=lgb_model.best_iteration)
        y_test_pred_masked = lgb_model.predict(self.test_Xtree[mask_test], num_iteration=lgb_model.best_iteration)
        rsme = np.sqrt(np.mean((y_train_pred_masked - self.train_ytree[mask_train]) ** 2))
        logger.info(f"  Train (lgbm_pred - ytree)       -> RSME: {rsme:.4f}")
        logger.info(f"LGB Prediction completed in {datetime.datetime.now() - startTime}.")
        
        # filter unimportant features
        # to be implemented

        # Return everything needed
        return {
            'lstm_model': lstm_model,
            'lgb_model': lgb_model,
            'y_test_pred_masked': y_test_pred_masked,
            'mask_train': mask_train,
            'mask_test': mask_test,
        }
        
    def __establish_weights(self) -> np.array:
        nSamples = self.train_Xtree.shape[0]
        mask_sparsing = np.random.rand(nSamples) <= 1e4/nSamples # for speedup
        
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
    
    def __df_analysis(self, y_test_pred: np.array, meta_pl: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        m = self.params['TreeTime_top_n']
        
        meta_pred_df = meta_pl.with_columns(
            pl.Series("prediction_ratio", y_test_pred)
        )
        
        # stoploss, to be implemented:
        """def cond(col):
            return col <= self.params['TreeTime_stoploss']
        
        cols = [f"target_close_at{i}" for i in range(1, self.idxDaysAfter + 1)]
        exprs = [
            pl.when(cond(pl.col(c)/pl.col("Close")))
            .then(pl.col(c))
            .otherwise(None)
            for c in cols[:-1]
        ] + [pl.col(cols[-1])]

        meta_pred_df = meta_pred_df.with_columns(
            (pl.coalesce(*exprs)).alias("result_close")
        ).with_columns(
            (pl.col('result_close')/pl.col('Close')).alias("result_ratio")
        )"""
        
        meta_pred_df = meta_pred_df.with_columns(
            (pl.col("target_ratio")).alias("result_ratio")
        )
        
        meta_pred_df = (
            meta_pred_df
            .sort(["date", "prediction_ratio"], descending=[False, True])
            .with_columns(
                pl.col("prediction_ratio")
                .rank(method="random", descending=True)
                .over("date")
                .alias("prediction_rank")
            ).filter(
                pl.col("prediction_rank") <= m
            )
        )
        
        meta_pred_df_perdate = (
            meta_pred_df.with_columns(
                (pl.col("prediction_ratio") - pl.col("target_ratio")).abs().alias("error")
            ).group_by("date").agg([
                pl.col("error").mean().alias("mean_error"),
                (pl.col("error")**2).mean().sqrt().alias("rmse_error"),
                pl.count().alias("n_entries"),
                pl.col("result_ratio").mean().alias("mean_result_ratio"),
            ])
        )
        
        return meta_pred_df, meta_pred_df_perdate
    
    def analyze(self, lstm_model = None, lgb_model = None, logger_disabled: bool = False) -> tuple[float, dict]:
        logger.disabled = logger_disabled
        
        # Run common pipeline in "analyze" mode
        data = self.pipeline(lstm_model = lstm_model, lgb_model = lgb_model)

        if data["lgb_model"] is not None:
            ModelAnalyzer.print_feature_importance_LGBM(data['lgb_model'], self.featureTreeNames, 15)
        
        # Additional analysis with test set
        y_test_pred_masked: np.array = data['y_test_pred_masked']
        mask_test: np.array = data['mask_test']
        
        meta_pred_df, meta_pred_df_perdate = self.__df_analysis(y_test_pred_masked, self.meta_pl_test.filter(mask_test))
        
        logger.info("Analyzing test set predictions:")
        logger.info(f"  Number of test dates: {len(self.test_dates)}")
        logger.info(f"  Ratio of test dates with choices: {meta_pred_df_perdate.shape[0] / len(self.test_dates):.4f}")
        logger.info(f"  Mean error per date {meta_pred_df_perdate['mean_error'].mean():.4f}")
        logger.info(f"  Mean RMSE error per date {meta_pred_df_perdate['rmse_error'].mean():.4f}")
        
        for _, test_date in enumerate(self.test_dates):
            logger.info(f"Analyzing test date: {test_date}")
            meta_pl_test_ondate: pl.DataFrame = (
                meta_pred_df
                .filter(pl.col("date") == test_date)
                .select(
                    ['date', 'ticker', 'Close']
                    +[f"target_close_at{i}" for i in range(1, self.idxDaysAfter + 1)]
                    +['prediction_ratio']
                    +['result_ratio']
                )
            )
            if meta_pl_test_ondate.is_empty():
                logger.error(f"No data available for test date {test_date}.")
                continue

            with pl.Config(ascii_tables=True):
                pl.Config.set_tbl_rows(15)
                pl.Config.set_tbl_cols(15)
                logger.info(f"DataFrame:\n{meta_pl_test_ondate}")
                
            logger.info(f"  P/L Ratio: {meta_pl_test_ondate['result_ratio'].mean():.4f}")
            logger.info(f"  Mean Prediction Ratio: {meta_pl_test_ondate['prediction_ratio'].mean():.4f}")
                
        res_ = meta_pred_df.group_by("date").agg([
            pl.col("result_ratio").mean().alias("mean_result_ratio"),
            pl.col("result_ratio").count().alias("n_entries"),
            pl.col("prediction_ratio").max().alias("max_pred_ratio"),
            pl.col("prediction_ratio").mean().alias("mean_pred_ratio"),
        ])
        
        res_pl = res_['mean_result_ratio'].mean()
        res_n = res_['n_entries'].sum()
        res_max_pred = res_['max_pred_ratio'].mean()
        res_mean_pred = res_['mean_pred_ratio'].mean()
        logger.info(f"Over all mean P/L Ratio: {res_pl:.4f}")
        logger.info(f"Over all mean prediction ratio: {res_mean_pred:.4f}")

        return (
            res_pl, 
            {
                'result': res_pl, 
                "n_entries": res_n, 
                "max_pred": res_max_pred, 
                "mean_pred": res_mean_pred
            }
        )

    def predict(self):
        # Run common pipeline in "analyze" mode
        data = self.pipeline()

        # Additional analysis with test set
        y_test_pred: np.array = data['y_test_pred_masked']

        # Top m analysis
        m = self.params['TreeTime_top_n']
        top_m_indices = np.flip(np.argsort(y_test_pred)[-m:])
        selected_df = self.meta_pl_test.with_columns(pl.Series("prediction_ratio", y_test_pred))[top_m_indices]
        
        selected_pred_values_reg = y_test_pred[top_m_indices]
        logger.info(f"Mean pred value of top {m}: {np.mean(selected_pred_values_reg)}")

        with pl.Config(ascii_tables=True):
            logger.info(f"DataFrame:\n{selected_df}")
        
        return (
            np.mean(selected_pred_values_reg)
        )