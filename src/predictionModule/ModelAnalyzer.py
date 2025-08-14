import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
from collections import Counter
from typing import Optional
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import lightgbm as lgb
from scipy import stats

import logging
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    def __init__(self):
        pass
    
    @staticmethod
    def print_label_distribution(*arrays):
        """
        Print the distribution of one or more label arrays.

        Parameters:
            *arrays (np.ndarray): One or more arrays of labels.
        """
        for idx, arr in enumerate(arrays, start=1):
            unique_vals, counts = np.unique(arr, return_counts=True)
            total = counts.sum()
            for val, count in zip(unique_vals, counts):
                freq = count / total
                logger.info(f"  Label {val}: Count = {count}, Frequency = {freq:.2f}")
            logger.info("")

    @staticmethod 
    def print_classification_metrics(y_known: np.ndarray, y_observer: np.ndarray, y_obs_proba: np.ndarray = None):
        """
        Print per-class accuracy, overall accuracy, log loss, and TPR, FPR, TNR, FNR for each class.

        Parameters:
            y_known (np.ndarray): known labels.
            y_observer (np.ndarray): observed labels.
            y_obs_proba (np.ndarray, optional): Predicted probabilities.
        """
        classes = np.unique(y_known)
        cm:np.array = confusion_matrix(y_known, y_observer, labels=classes)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

        overall_acc = accuracy_score(y_known, y_observer)
        logger.info(f"\n  Overall Accuracy: {overall_acc:.2f}")

        if y_obs_proba is not None:
            ll = log_loss(y_known, y_obs_proba)
            logger.info(f"  Log Loss: {ll:.4f}")

        logger.info("\n  Metrics per Class:")
        for i, cls in enumerate(classes):
            TP = cm[i, i]
            FN = cm[i].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FP + FN)

            TPR = TP / (TP + FN) if (TP + FN) else 0
            FPR = FP / (FP + TN) if (FP + TN) else 0
            TNR = TN / (TN + FP) if (TN + FP) else 0
            FNR = FN / (FN + TP) if (FN + TP) else 0

            logger.info(f"    Class {cls}:")
            logger.info(f"      TPR: {TPR:.2f}, FPR: {FPR:.2f}, TNR: {TNR:.2f}, FNR: {FNR:.2f}") 
            
        logger.info("")
    
    @staticmethod
    def print_feature_importance_LGBM(lgbModel: lgb.LGBMClassifier, featureColumnNames: list[str], n_feature: int = 20):
        importances = lgbModel.feature_importances_
        
        feature_importances = pd.DataFrame({
            'Feature': featureColumnNames,
            'Importance': importances
        })
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        top_n = min(n_feature, feature_importances.shape[0])
        top_features = feature_importances.head(top_n).reset_index(drop=True)
        top_features.index += 1  # Start ranking at 1
        top_features.index.name = 'Rank'
        top_features['Importance'] = top_features['Importance'].apply(lambda x: f"{x:.4f}")
        logger.info(f"Top {top_n} Feature Importances:")
        logger.info(top_features.to_string())
    
    @staticmethod
    def print_feature_importance_LGBM(lgbModel: lgb.Booster, featureColumnNames: list[str], n_feature: int = 20):
        importances = lgbModel.feature_importance()
        
        feature_importances = pd.DataFrame({
            'Feature': featureColumnNames,
            'Importance': importances
        })
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        top_n = min(n_feature, feature_importances.shape[0])
        top_features = feature_importances.head(top_n).reset_index(drop=True)
        top_features.index += 1  # Start ranking at 1
        top_features.index.name = 'Rank'
        top_features['Importance'] = top_features['Importance'].apply(lambda x: f"{x:.4f}")
        logger.info(f"Top {top_n} Feature Importances:")
        logger.info(top_features.to_string())
        
    @staticmethod
    def log_test_result_perdate(test_df: pl.DataFrame, test_dates: list[str], last_col: str | None = None):
        """Log test results for each date in the test_dates list.

        The dataframe needs the columns ['date', 'ticker', 'Close', 'prediction_ratio', last_col].
        Is last_col is None we assume it is for prediction.

        Args:
            test_df (pl.DataFrame): The DataFrame containing test results.
            test_dates (list[str]): The list of test dates to analyze.
        """
        if last_col not in test_df.columns:
            logger.warning(f"Last column '{last_col}' not found in test_df.")
            last_col = None

        for test_date in test_dates:
            logger.info(f"Analyzing test date: {test_date}")

            # Filter meta dataframe on test date
            select_cols = ['date', 'ticker', 'Close','prediction_ratio']
            select_cols = select_cols + [last_col] if last_col else select_cols
            res_df_ondate: pl.DataFrame = (
                test_df
                .filter(pl.col("date") == test_date)
                .select(select_cols)
            )
            if res_df_ondate.is_empty():
                logger.error(f"No data available for test date {test_date}.")
                continue

            # Print result on test date
            with pl.Config(ascii_tables=True, tbl_rows=20, tbl_cols=20):
                logger.info(f"DataFrame:\n{res_df_ondate}")
                
            if last_col is not None:
                logger.info(f"  P/L Ratio: {res_df_ondate[last_col].mean():.4f}")
            logger.info(f"  Mean Prediction Ratio: {res_df_ondate['prediction_ratio'].mean():.4f}")
            
    @staticmethod
    def log_test_result_overall(test_df: pl.DataFrame, last_col: str | None = None):
        agg_exprs = [
            pl.col("prediction_ratio").max().alias("max_pred"),  # this is also .first()
            pl.col("prediction_ratio").mean().alias("mean_pred"),
        ]
        if last_col is not None:
            agg_exprs.extend([
                pl.col(last_col).mean().alias("mean_res"),
                pl.col(last_col).first().alias("top_res"),
                pl.col(last_col).count().alias("n_entries")
            ])
        test_df_perdate = test_df.group_by("date").agg(agg_exprs)

        pred_meanmean = test_df_perdate['mean_pred'].mean()
        pred_meanlast = test_df_perdate['mean_pred'].last()
        pred_toplast = test_df_perdate['max_pred'].last()
        logger.info(f"Over all mean prediction ratio: {pred_meanmean:.4f}")
        logger.info(f"Over all top last prediction ratio: {pred_toplast:.4f}")
        logger.info(f"Over all last mean prediction ratio: {pred_meanlast:.4f}")
        
        if last_col is not None:
            res_meanmean = test_df_perdate['mean_res'].mean()
            res_meanlast = test_df_perdate['mean_res'].last()
            res_topmean = test_df_perdate['top_res'].mean()
            res_toplast = test_df_perdate['top_res'].last()
            res_sum_n = test_df_perdate['n_entries'].sum()
            
            logger.info(f"Over all mean P/L Ratio: {res_meanmean:.4f}")
            logger.info(f"Over all top mean P/L Ratio: {res_topmean:.4f}")
            logger.info(f"Over all top last P/L Ratio: {res_toplast:.4f}")
            logger.info(f"Over all mean last P/L Ratio: {res_meanlast:.4f}")
            logger.info(f"Over all number of entries: {res_sum_n}")
            
    @staticmethod
    def log_test_result_multiple(df_list: list[pl.DataFrame], last_col: str) -> pl.DataFrame:
        results = []
        for df in df_list:
            end_train_date = df.select('date').min().item()
            end_test_date = df.select('date').max().item()
            df_perdate = df.group_by("date").agg([
                pl.col(last_col).mean().alias("mean_res"),
                pl.col(last_col).first().alias("top_res"),
                pl.col(last_col).count().alias("n_entries"),
                pl.col("prediction_ratio").max().alias("max_pred"),
                pl.col("prediction_ratio").mean().alias("mean_pred"),
            ])
            results.append(
                {
                    "end_train_date": end_train_date,
                    "end_test_date": end_test_date,
                    "res_meanmean": df_perdate['mean_res'].mean(),
                    "res_toplast": df_perdate['top_res'].last(),
                    "res_meanlast": df_perdate['mean_res'].last(),
                    "n_entries": df_perdate['n_entries'].sum(),
                    "pred_toplast": df_perdate['max_pred'].last(),
                    "pred_meanmean": df_perdate['mean_pred'].mean(),
                    "pred_meanlast": df_perdate['mean_pred'].last(),
                }
            )

        results_df = pl.DataFrame(results).sort("end_train_date")
        with pl.Config(ascii_tables=True, tbl_rows=-1, tbl_cols=-1):
            logger.info(results_df)

        logger.info(f"Mean over meanmean returns over all cutoffs: {results_df['res_meanmean'].mean()}")
        logger.info(f"Mean over toplast returns over all cutoffs: {results_df['res_toplast'].mean()}")
        logger.info(f"Mean over meanlast returns over all cutoffs: {results_df['res_meanlast'].mean()}")
        logger.info(f"Mean over meanmean predictions over all cutoffs: {results_df['pred_meanmean'].mean()}")
        logger.info(f"Mean over toplast predictions over all cutoffs: {results_df['pred_toplast'].mean()}")
        logger.info(f"Mean over meanlast predictions over all cutoffs: {results_df['pred_meanlast'].mean()}")
        logger.info(f"Total entries over all cutoffs: {results_df['n_entries'].sum()}")

        if len(results_df) > 3:
            # Quantiles
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                pred_meanmean = np.quantile(results_df['pred_meanmean'].to_numpy(), q)
                pred_meanlast = np.quantile(results_df['pred_meanlast'].to_numpy(), q)
                pred_toplast = np.quantile(results_df['pred_toplast'].to_numpy(), q)
                logger.info(f"Quantile {q:.1f}:")
                logger.info(f"  Pred meanmean: {pred_meanmean:.4f}")
                logger.info(f"  Pred meanlast: {pred_meanlast:.4f}")
                logger.info(f"  Pred toplast: {pred_toplast:.4f}")


            # Conditional Means
            with pl.Config(ascii_tables=True, tbl_rows=-1, tbl_cols=-1):
                logger.info(
                    "Mean over results filtered by 0.5 quantile prediction meanmean: "
                    f"{results_df.filter(pl.col('pred_meanmean') > pl.col('pred_meanmean').quantile(0.5)).mean()}"
                )
                logger.info(
                    f"Mean over results filtered by 0.5 quantile prediction meanlast: "
                    f"{results_df.filter(pl.col('pred_meanlast') > pl.col('pred_meanlast').quantile(0.5)).mean()}"
                )
                logger.info(
                    f"Mean over results filtered by 0.5 quantile prediction toplast: "
                    f"{results_df.filter(pl.col('pred_toplast') > pl.col('pred_toplast').quantile(0.5)).mean()}"
                )
        return results_df
