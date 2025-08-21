import numpy as np
import polars as pl
import logging
import datetime

from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.mathTools.DistributionTools import DistributionTools
from src.predictionModule.LoadupSamples import LoadupSamples
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.predictionModule.WeightSamples import WeightSamples

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
        if self.params['TreeTime_WeightSamples_run']:
            logger.info("Establishing weights for TreeTime features...")
            ws = WeightSamples(
                Xtree_train=self.train_Xtree,
                ytree_train=self.train_ytree,
                Xtree_test=self.test_Xtree,
                treenames=self.featureTreeNames,
                params=self.params
            )
            self.tree_weights = ws.establish_weights()
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
    
    def __get_top_tickers(self, y_test_pred: np.ndarray, meta_pl: pl.DataFrame) -> pl.DataFrame:
        m = self.params['TreeTime_top_n']
        
        meta_pred_df = meta_pl.with_columns(
            pl.Series("prediction_ratio", y_test_pred)
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

        return meta_pred_df
    
    def __df_analysis(self, y_test_pred: np.ndarray, meta_pl: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        m = self.params['TreeTime_top_n']
        
        meta_pl = meta_pl.with_columns(
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
        
        meta_pl = meta_pl.with_columns(
            (pl.col("target_ratio")).alias("result_ratio")
        )
        
        meta_pl_filtered = (
            meta_pl
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
        
        meta_pl_filtered_perdate = (
            meta_pl_filtered.with_columns(
                (pl.col("prediction_ratio") - pl.col("target_ratio")).abs().alias("error")
            ).group_by("date").agg([
                pl.col("error").mean().alias("mean_error"),
                (pl.col("error")**2).mean().sqrt().alias("rmse_error"),
                pl.count().alias("n_entries"),
                pl.col("result_ratio").mean().alias("mean_result_ratio"),
            ])
        )
        
        return meta_pl_filtered, meta_pl_filtered_perdate
    
    def analyze(self, lstm_model = None, lgb_model = None, logger_disabled: bool = False) -> tuple[float, dict]:
        logger_config = logger.disabled
        logger.disabled = logger_disabled
        
        # Run common pipeline in "analyze" mode
        data = self.pipeline(lstm_model = lstm_model, lgb_model = lgb_model)

        if data["lgb_model"] is not None:
            ModelAnalyzer.print_feature_importance_LGBM(data['lgb_model'], self.featureTreeNames, 15)
        
        # Additional analysis with test set
        y_test_pred_masked: np.ndarray = data['y_test_pred_masked']
        mask_test: np.ndarray = data['mask_test']
        
        res_df, res_df_perdate_err = (
            self.__df_analysis(
                y_test_pred_masked, 
                self.meta_pl_test.filter(pl.Series(mask_test))
            )
        )
        
        logger.info("Analyzing test set predictions:")
        logger.info(f"  Number of test dates: {len(self.test_dates)}")
        logger.info(f"  Ratio of test dates with choices: {res_df_perdate_err.shape[0] / len(self.test_dates):.4f}")
        logger.info(f"  Mean error per date {res_df_perdate_err['mean_error'].mean():.4f}")
        logger.info(f"  Mean RMSE error per date {res_df_perdate_err['rmse_error'].mean():.4f}")

        ModelAnalyzer.log_test_result_perdate(res_df, self.test_dates, last_col = "target_ratio")

        res_df_perdate = res_df.group_by("date").agg([
            pl.col("target_ratio").mean().alias("mean_res"),
            pl.col("target_ratio").first().alias("top_res"),
            pl.col("target_ratio").count().alias("n_entries"),
            pl.col("prediction_ratio").max().alias("max_pred"),  # this is also .first()
            pl.col("prediction_ratio").mean().alias("mean_pred"),
        ])

        ModelAnalyzer.log_test_result_overall(res_df, last_col = "target_ratio")

        logger.disabled = logger_config
        return (
            res_df_perdate['mean_pred'].mean(), 
            {
                "df_pred_res": res_df.select(['date', 'ticker', 'Close', 'prediction_ratio', 'target_ratio']),
                "df_pred_res_perdate": res_df_perdate.select(['date', 'n_entries', 'max_pred', 'mean_pred', 'top_res', 'mean_res'])
            }
        )

    def predict(self, lstm_model = None, lgb_model = None, logger_disabled: bool = False) -> tuple[float, dict]:
        logger_config = logger.disabled
        logger.disabled = logger_disabled
        
        # Run common pipeline in "analyze" mode
        data = self.pipeline(lstm_model = lstm_model, lgb_model = lgb_model)

        if data["lgb_model"] is not None:
            ModelAnalyzer.print_feature_importance_LGBM(data['lgb_model'], self.featureTreeNames, 15)
        
        y_test_pred_masked: np.ndarray = data['y_test_pred_masked']
        mask_test: np.ndarray = data['mask_test']

        res_df = self.__get_top_tickers(y_test_pred_masked, self.meta_pl_test.filter(mask_test))

        if res_df.is_empty():
            raise ValueError("No valid predictions sampled.")

        ModelAnalyzer.log_test_result_perdate(res_df, self.test_dates, last_col = None)

        res_df_perdate = res_df.group_by("date").agg([
            pl.col("prediction_ratio").count().alias("n_entries"),
            pl.col("prediction_ratio").mean().alias("mean_pred"),
            pl.col("prediction_ratio").max().alias("max_pred"),
        ])

        ModelAnalyzer.log_test_result_overall(res_df, last_col = None)

        logger.disabled = logger_config

        return res_df_perdate['mean_pred'].mean(), {
            "df_pred_res": res_df.select(['date', 'ticker', 'Close', 'prediction_ratio']),
            "df_pred_res_perdate": res_df_perdate.select(['date', 'n_entries', 'max_pred', 'mean_pred']),
        }
