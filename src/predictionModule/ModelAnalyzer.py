import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from src.predictionModule.IML import IML
import lightgbm as lgb
from scipy import stats

import logging
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    def __init__(self, module = None):
        """
        Initialize the ModelAnalyzer with a trained model instance.

        Parameters:
            model (FourierML): An instance of the FourierML class containing trained models and data.
        """
        self.module: IML = module
        sns.set(style="whitegrid")

    def plot_label_distribution(self, label_data: Optional[np.ndarray] = None, title: str = 'Percentage Distribution of Labels', palette: str = 'viridis'):
        """
        Plot the distribution of class labels with percentage annotations.

        Parameters:
            label_data (np.ndarray, optional): Array of labels to plot. Defaults to model's y_train.
            title (str): Title of the plot.
            palette (str): Seaborn color palette.
        
        Returns:
            float: Sum of squared percentages.
        """
        if label_data is None:
            label_data = self.module.y_train
        
        plt.figure(figsize=(10,6))
        total = len(label_data)
        ax = sns.countplot(x=label_data, palette=palette)

        # Add percentage labels on top of each bar
        percentages = []
        for p in ax.patches:
            perc = '{:.1f}%'.format(100 * p.get_height() / total)
            percentages.append(p.get_height() / total)
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 1, perc, ha="center") 
        
        plt.title(title)
        plt.xlabel('Class Labels')
        plt.ylabel('Frequency')
        plt.show()
        
        percentage_sum_sq = np.sum(np.array(percentages) ** 2)
        print(f"Sum of squared percentages: {percentage_sum_sq:.4f}")
        return percentage_sum_sq
    
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
    def print_feature_importance_LGBM(model: IML, n_feature: int = 50):
        importances = model.LGBMModel.feature_importances_
        
        feature_importances = pd.DataFrame({
            'Feature': model.featureColumnNames,
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


    def plot_per_class_accuracy(self, y_val: np.ndarray, y_pred: np.ndarray, test_acc: float, test_loss: float):
        """
        Calculate and plot per-class accuracy.

        Parameters:
            y_val (np.ndarray): True labels for validation set.
            y_pred (np.ndarray): Predicted labels for validation set.
            test_acc (float): Overall test accuracy.
            test_loss (float): Overall test log loss.
        """
        most_common_class = Counter(y_val).most_common(1)[0][0]
        print(f"Most common class in validation set: {most_common_class}")

        # Calculate per-class accuracy
        classes = np.unique(y_val)
        per_class_accuracy = {}
        for cls in classes:
            idx = (y_val == cls)
            if np.sum(idx) > 0:
                per_class_accuracy[cls] = accuracy_score(y_val[idx], y_pred[idx])
            else:
                per_class_accuracy[cls] = np.nan  # Handle classes with no samples

        # Plot per-class accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(per_class_accuracy.keys(), per_class_accuracy.values(), color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Print overall metrics
        print(f'\nTest Accuracy: {test_acc:.4f}')
        print(f'Test Log Loss: {test_loss:.4f}')

    def plot_feature_importance(self, max_num_features: int = 20, height: float = 0.5, save_path: str = 'feature_importance.png'):
        """
        Plot and save feature importance for a specified model.

        Parameters:
            model_name (str): Name of the model attribute in the IML instance.
            max_num_features (int): Maximum number of top features to display.
            height (float): Height of each feature importance bar.
            save_path (str): File path to save the plot.
        """
        model = getattr(self.module, 'XGBoostModel', None)
        if model is None:
            raise ValueError(f"Model 'XGBoostModel' not found in the provided model instance.")

        plt.figure(figsize=(10, 8))
        xgb.plot_importance(model, max_num_features=max_num_features, height=height, show_values=False)
        plt.title(f'Feature Importance for XGBoostModel')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        print(f"Feature importance plot saved to {save_path}")

    def plot_lstm_absolute_diff_histogram(self, 
                                          bins: int = 50, color: str = 'skyblue', 
                                          edgecolor: str = 'black', alpha: float = 0.7, 
                                          title: str = 'Histogram of Absolute Differences Between Predictions and Actual Values'):
        """
        Plot histograms of absolute and relative differences between LSTM predictions and actual values.

        Parameters:
            bins (int): Number of bins for the histogram.
            color (str): Color of the histogram bars.
            edgecolor (str): Edge color of the histogram bars.
            alpha (float): Transparency level of the histogram bars.
            title (str): Title of the histogram.
        """
        # Check if the LSTM model is trained
        if self.module.LSTMModel is None:
            raise ValueError("LSTM model is not trained or not available.")
        
        # Prepare and scale the test data
        X_test: np.array = self.module.X_test_timeseries
        y_test: np.array = self.module.y_test_timeseries  # shape (:,1)
        
        num_samples, timesteps, num_features = X_test.shape
        
        # Flatten X_test to 2D for scaling
        X_test_flat = X_test.reshape(num_samples, -1)
        X_test_scaled_flat = self.module.scaler_X.transform(X_test_flat)
        X_test_scaled = X_test_scaled_flat.reshape(num_samples, timesteps, num_features)
        
        # Scale y_test
        y_test_scaled = self.module.scaler_y.transform(y_test)
        
        # Predict with LSTM using scaled X_test
        predictions_scaled = self.module.LSTMModel.predict(X_test_scaled)
        
        # Inverse scale the predictions
        predictions = self.module.scaler_y.inverse_transform(predictions_scaled)
        
        # Compute the absolute and relative differences
        abs_diff = np.abs(predictions.flatten() - y_test.flatten())
        rel_diff = abs_diff / np.abs(y_test.flatten())
        
        # Plot the histogram of absolute differences
        plt.figure(figsize=(10,6))
        plt.hist(abs_diff, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
        plt.title('Histogram of Absolute Differences Between Predictions and Actual Values')
        plt.xlabel('Absolute Difference')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        
        # Plot the histogram of relative differences
        plt.figure(figsize=(10,6))
        plt.hist(rel_diff, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
        plt.title('Histogram of Relative Differences Between Predictions and Actual Values')
        plt.xlabel('Relative Difference')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def print_model_results(pred: list, ret: list):
        logger.info(f"Resulting returns: {ret}")
        logger.info(f"Resulting returns mean: {np.mean([x for x in ret if x is not None])}")
        logger.info(f"Resulting returns variance: {np.var([x for x in ret if x is not None])}")
        logger.info("")
        logger.info(f"Resulting predictions: {pred}")
        logger.info(f"Resulting predictions mean: {np.mean([x for x in pred if x is not None])}")
        logger.info(f"Resulting predictions variance: {np.var([x for x in pred if x is not None])}")
        logger.info("")

        logger.info("Statictical Analysis res_pred to res_return")
        # print statistic like regression for res_pred to res_return
        # Calculate the correlation coefficient and the p-value
        correlation, p_value = stats.pearsonr(pred, ret)

        # Log the results
        logger.info(f"Correlation coefficient: {correlation}")

        # Perform a linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(pred, ret)

        # Log the regression results
        logger.info(f"Linear regression slope: {slope}")
        logger.info(f"Linear regression intercept: {intercept}")
        logger.info(f"R-squared: {r_value**2}")
        logger.info(f"P-value: {p_value}")
        logger.info(f"Standard error: {std_err}")

        # Variance of the residuals
        residuals = np.array(ret) - (slope * np.array(pred) + intercept)
        residuals_variance = np.var(residuals)
        logger.info(f"Variance of the residuals: {residuals_variance}") 