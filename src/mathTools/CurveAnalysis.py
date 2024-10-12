from typing import Dict
import pandas as pd
import numpy as np
from scipy.stats import linregress

class CurveAnalysis:

    def __init__():
        pass

    @staticmethod
    def thirdDegreeFit(priceArray, ticker: str) -> Dict:
        x = np.arange(len(priceArray))
        y = priceArray

        # Perform cubic polynomial regression with covariance
        coeff, covariance = np.polyfit(x, y, 3, cov=True)
            # x[0]**3 * coeff[0]+...+x[0] * coeff[2]+coeff[3]  approx y[0]

        # Calculate the fitted y-values
        y_fit = np.polyval(coeff, x)

        # Compute residuals
        residuals = y - y_fit

        # Calculate variance of residuals (using sample variance, ddof=1)
        variance = np.var(residuals, ddof=1)

        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Standard errors of the coefficients are the square roots of the diagonal elements of the covariance matrix
        std_err = np.sqrt(np.diag(covariance))

        fit_results = {
            'Ticker': ticker,
            'Coefficients': coeff.tolist(),  # Convert to list for JSON serialization
            'Std_Errors': std_err.tolist(),
            'R_Squared': r_squared,
            'Variance': variance
        }

        return fit_results

    @staticmethod
    def lineFit(priceArray, ticker: str) -> Dict:
        x = np.arange(len(priceArray))
        y = priceArray

        # Perform linear regression using scipy
        regression = linregress(x, y)

        # Calculate the fitted y-values
        y_fit = regression.slope * x + regression.intercept

        # Compute residuals
        residuals = y - y_fit

        # Calculate variance of residuals (using sample variance, ddof=1)
        variance = np.var(residuals, ddof=1)
    
        fit_results = {
            'Ticker': ticker,
            'Slope': regression.slope,
            'Intercept': regression.intercept,
            'R_Value': regression.rvalue,
            'P_Value': regression.pvalue,
            'Std_Err': regression.stderr,
            'Variance': variance
        }

        return fit_results

    @staticmethod
    def lineFitThroughFirstPoint(priceArray, ticker: str) -> Dict:
        x = np.arange(len(priceArray))

        y = priceArray

        # Fit line through the first data point and minimize residuals
        x0 = x[0]
        y0 = y[0]
        dx = x - x0
        dy = y - y0

        denominator = np.sum(dx * dx)
        if denominator == 0:
            raise ValueError("Denominator is zero; all x values are the same.")
        
        # Calculate the slope (m) and intercept (c)
        m = np.sum(dx * dy) / denominator
        c = y0 - m * x0

        y_pred = m * x + c
        residuals = y-y_pred
        
        mean_y_pred = np.mean(y_pred)
        if mean_y_pred == 0:
            mean_y_pred = np.finfo(float).eps  # Smallest representable float
        variance = np.var(residuals / mean_y_pred)

        return {
            'Ticker': ticker,
            'Slope': m,
            'Variance': variance
        }