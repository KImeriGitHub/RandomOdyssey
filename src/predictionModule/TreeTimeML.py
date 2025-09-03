import numpy as np
import polars as pl
import logging
import datetime

import scipy

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

        "TreeTime_LSTM_days_to_train": 300,
        "TreeTime_FilterSamples_method": "taylor",

        "LoadupSamples_time_inc_factor": 10,
        "LoadupSamples_tree_scaling_standard": True,
        "LoadupSamples_time_scaling_stretch": True,
    }

    def __init__(
            self, 
            train_start_date: datetime.date,
            test_dates: list[datetime.date],
            treegroup: str,
            timegroup: str,
            params: dict = None,
            loadup: LoadupSamples = None
        ):
        
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.treegroup = treegroup
        self.timegroup = timegroup
        self.train_start_date = train_start_date
        self.test_dates = test_dates
        
        self.min_test_date = min(self.test_dates)
        self.max_test_date = max(self.test_dates)
        
        # Assign parameters to instance variables
        if loadup is None or not isinstance(loadup, LoadupSamples):
            ls = LoadupSamples(
                train_start_date=self.train_start_date,
                test_dates=self.test_dates,
                treegroup=self.treegroup,
                timegroup=self.timegroup,
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

        self.mask_train = np.ones(self.train_Xtree.shape[0], dtype=bool)
        self.mask_test = np.ones(self.test_Xtree.shape[0], dtype=bool)
        
    def pipeline(self, lstm_model = None, lgb_model = None) -> dict:
        """
        Common pipeline steps shared by both analyze() and predict().
        Returns a dictionary of all relevant masked data, trained model, and predictions.
        """
        # Filter samples
        fs_pre = FilterSamples(
            Xtree_train = self.train_Xtree,
            ytree_train = self.train_ytree,
            treenames   = self.featureTreeNames,
            Xtree_test  = self.test_Xtree,
            meta_train  = self.meta_pl_train,
            meta_test   = self.meta_pl_test,
            ytree_test  = self.test_ytree,
            params      = self.params
        )

        # Filter categorical
        cat_mask_train, cat_mask_test = fs_pre.categorical_masks()
        self.mask_train &= cat_mask_train
        self.mask_test &= cat_mask_test

        # run LSTM to add time prediction to tabular data
        logger.info("Running LSTM to add time prediction to tree data...")
        if self.params['TreeTime_run_lstm']:
            days_to_train = self.params["TreeTime_LSTM_days_to_train"]
            mask_dates_reduced = fs_pre.get_recent_training_mask(days_to_train)
            self.__run_time_to_tree_addition(lstm_model, mask_dates_reduced)

        # Filter through lincomb strategy
        fs = FilterSamples(
            Xtree_train = self.train_Xtree[self.mask_train],
            ytree_train = self.train_ytree[self.mask_train],
            treenames   = self.featureTreeNames,
            Xtree_test  = self.test_Xtree[self.mask_test],
            meta_train  = self.meta_pl_train.filter(pl.Series(self.mask_train)),
            meta_test   = self.meta_pl_test.filter(pl.Series(self.mask_test)),
            ytree_test  = self.test_ytree[self.mask_test],
            params      = self.params
        )
        if self.params['TreeTime_FilterSamples_method'] == "taylor":
            fs_mask_train, fs_mask_test = fs.taylor_feature_masks()
        if self.params['TreeTime_FilterSamples_method'] == "lincomb":
            fs_mask_train, fs_mask_test = fs.lincomb_masks()
        if self.params['TreeTime_FilterSamples_method'] not in ["taylor", "lincomb"]:
            logger.warning(f"Unknown FilterSamples method: {self.params['TreeTime_FilterSamples_method']}")
            fs_mask_train = np.ones(self.mask_train.sum(), dtype=bool)
            fs_mask_test = np.ones(self.mask_test.sum(), dtype=bool)
        self.mask_train[self.mask_train] = fs_mask_train
        self.mask_test[self.mask_test]   = fs_mask_test
        
        ## Establish weights for Tree
        startTime  = datetime.datetime.now()
        if self.params['TreeTime_WeightSamples_run']:
            logger.info("Establishing weights for TreeTime features...")
            ws = WeightSamples(
                Xtree_train=self.train_Xtree[self.mask_train],
                ytree_train=self.train_ytree[self.mask_train],
                Xtree_test=self.test_Xtree[self.mask_test],
                treenames=self.featureTreeNames,
                params=self.params
            )
            self.tree_weights = np.ones(self.train_Xtree.shape[0], dtype=np.float64)
            masked_weights = ws.establish_weights()
            self.tree_weights[self.mask_train] = masked_weights
            logger.info(f"Weights established in {datetime.datetime.now() - startTime}.")
        else:
            logger.info("Skipping TreeTime feature weights establishment.")
            self.tree_weights = np.ones(self.train_Xtree.shape[0], dtype=np.float64)
        
        # LGB model
        if lgb_model is None:
            startTime  = datetime.datetime.now()
            mm = MachineModels(self.params)
            days_to_train_LGB = self.params["TreeTime_LGB_days_to_train"]
            mask_dates_reduced = fs.get_recent_training_mask(days_to_train_LGB)
            lgb_model, lgb_res_dict = mm.run_LGB(
                X_train=self.train_Xtree[days_to_train_LGB & self.mask_train],
                y_train=self.train_ytree[days_to_train_LGB & self.mask_train],
                X_test=self.test_Xtree[self.mask_test],
                y_test=self.test_ytree[self.mask_test],
                weights=self.tree_weights[days_to_train_LGB & self.mask_train],
            )
        
        # LGB Predictions
        startTime  = datetime.datetime.now()
        y_train_pred_masked = lgb_model.predict(self.train_Xtree, num_iteration=lgb_model.best_iteration)
        y_test_pred_masked = lgb_model.predict(self.test_Xtree, num_iteration=lgb_model.best_iteration)
        rmse = np.sqrt(np.mean((y_train_pred_masked - self.train_ytree) ** 2))
        logger.info(f"  Train (lgbm_pred - ytree)       -> RMSE: {rmse:.4f}")
        logger.info(f"  LGB completed in {datetime.datetime.now() - startTime}.")

        # Return everything needed
        return {
            'lstm_model': lstm_model,
            'lgb_model': lgb_model,
            'y_test_pred_masked': y_test_pred_masked,
            'mask_train': self.mask_train,
            'mask_test': self.mask_test,
        }
    
    def __run_time_to_tree_addition(self, lstm_model, mask_days_reduced) -> None:
        """
        Runs LSTM to generate feature(s) to add to the tree data.
        """
        mm = MachineModels(self.params)

        starttime = datetime.datetime.now()
        device = "cuda"
        lstm_model, res_dict = mm.run_LSTM_torch(
            self.train_Xtime[mask_days_reduced & self.mask_train], 
            self.train_ytime[mask_days_reduced & self.mask_train], 
            device=device
        )
        preds_train = mm.predict_LSTM_torch(lstm_model, self.train_Xtime, batch_size=self.params["LSTM_batch_size"], device=device)
        preds_test = mm.predict_LSTM_torch(lstm_model, self.test_Xtime, batch_size=self.params["LSTM_batch_size"], device=device)
        endtime = datetime.datetime.now()

        logger.info(f"  LSTM RSME: {res_dict['val_rmse']*2/self.params['LoadupSamples_time_inc_factor']:.4f}")
        logger.info(f"  LSTM completed in {endtime - starttime}.")
        filtered_train = (
            self.train_ytree
                [self.mask_train]
                [
                    preds_train[self.mask_train] 
                        >= np.quantile(preds_train[self.mask_train], self.params["FilterSamples_q_up"])
                ]
        )
        logger.info(f"  Result of quantile {self.params['FilterSamples_q_up']:.2f}")
        logger.info(f"    Train set (gmean, unreduced): {scipy.stats.gmean(filtered_train):.4f}")
    
        ## Add LSTM predictions to the tree test set
        train_std = np.std(preds_train)
        test_std = 1.0 if np.std(preds_test) < 1e-6 else np.std(preds_test[self.mask_test])
        self.train_Xtree = np.hstack((self.train_Xtree, ((preds_train-1.0)/train_std).reshape(-1, 1)))
        self.test_Xtree = np.hstack((self.test_Xtree, ((preds_test-1.0)/test_std).reshape(-1, 1)))
        if isinstance(self.featureTreeNames, list):
            self.featureTreeNames.append("LSTM_Prediction")
        elif isinstance(self.featureTreeNames, np.ndarray):
            self.featureTreeNames = np.append(self.featureTreeNames, "LSTM_Prediction")

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
            res_df_perdate['mean_pred'].last(), 
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
