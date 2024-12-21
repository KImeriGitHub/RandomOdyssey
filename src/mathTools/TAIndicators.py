import numpy as np
import pandas as pd
import polars as pl
from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler
import warnings

class TAIndicators():
    """
    A class to compute technical analysis (TA) indicators and perform scaling operations on them.

    Attributes:
        tacolumns_to_minmax (list): List of TA indicator columns to be scaled using Min-Max scaling.
        tacolumns_to_keep (list): List of TA indicator columns to be kept for analysis.
        columns_to_del (list): List of columns to delete from the DataFrame.
    """
    
    tacolumns_to_minmax = [
        'Volume','volume_adi','volume_obv','volume_cmf','volume_fi','volume_em','volume_sma_em','volume_vpt','volume_mfi','volume_nvi',
        'volatility_bbp','volatility_kcw','volatility_kcp','volatility_dcw','volatility_dcp','volatility_atr','volatility_ui','trend_macd','trend_macd_signal',
        'trend_macd_diff','trend_vortex_ind_neg','trend_vortex_ind_diff','trend_trix','trend_mass_index','trend_dpo','trend_kst','trend_stc','trend_adx','trend_adx_pos','trend_adx_neg',
        'trend_cci','trend_aroon_up','trend_aroon_down','trend_aroon_ind','momentum_stoch_rsi','momentum_stoch_rsi_k','momentum_stoch_rsi_d','momentum_tsi',
        'momentum_uo','momentum_stoch','momentum_stoch_signal','momentum_wr','momentum_ao','momentum_roc','momentum_ppo','momentum_ppo_signal',
        'momentum_ppo_hist', 'volatility_bbhi','others_dlr','trend_psar_down_indicator','volatility_bbw','trend_kst_sig','trend_visual_ichimoku_b',
        'others_dr','volatility_kcli','volatility_bbli','trend_vortex_ind_pos','trend_psar_up_indicator','momentum_pvo_hist',
        'volatility_kchi','trend_visual_ichimoku_a','momentum_pvo_signal','momentum_rsi','trend_kst_diff','momentum_pvo', 'others_cr',
    ]
    tacolumns_to_keep = ['Open','High','Low','Close', 'trend_sma_slow', 'trend_ichimoku_conv', 'volatility_bbl', 'volatility_bbm', 
        'volatility_dcm', 'volatility_kcc', 'trend_psar_down', 'trend_sma_fast', 
        'volatility_kch', 'trend_ichimoku_base', 'volume_vwap', 'trend_ichimoku_a', 
        'trend_ema_slow', 'volatility_kcl', 'momentum_kama', 'trend_ema_fast', 'volatility_dcl', 'trend_psar_up', 
        'volatility_dch', 'trend_ichimoku_b', 'volatility_bbh'
    ]

    columns_to_del = ['Adj Close', 'Date']
        
    def __init__(self, df):
        """
        Initialize the TAIndicators class with a DataFrame.
        Adds all technical analysis features to the DataFrame.

        Parameters:
            df (pd.DataFrame or pl.DataFrame): Input DataFrame containing OHLCV data.
        """
        self.df = df
        if isinstance(df, pd.DataFrame):
            # Add all TA features to the pandas DataFrame
            self.tadata = add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
        elif isinstance(df, pl.DataFrame):
            # Convert polars DataFrame to pandas and add TA features
            self.tadata = add_all_ta_features(
                df.to_pandas(), open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
        else:
            # Raise an error if df is neither pandas nor polars DataFrame
            raise TypeError("Input df must be a pandas DataFrame or polars DataFrame.")

        # Delete specified columns if they exist
        self.tadata.drop(columns=self.columns_to_del, inplace=True, errors='ignore')

    
    def scale_MinMax(self) -> pd.DataFrame:
        """
        Scale selected technical indicators using Min-Max scaling.

        Returns:
            pd.DataFrame: DataFrame containing the scaled indicators.
        """
        # Check if all columns to be scaled are present
        missing_columns = [col for col in self.tacolumns_to_minmax if col not in self.tadata.columns]
        if missing_columns:
            warnings.warn(f"The following columns are missing in tadata and will be skipped: {missing_columns}")
            # Use only the columns that are present
            columns_to_scale = [col for col in self.tacolumns_to_minmax if col in self.tadata.columns]
        else:
            columns_to_scale = self.tacolumns_to_minmax

        if not columns_to_scale:
            raise ValueError("No columns available for scaling.")

        # Perform Min-Max scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.tadata[columns_to_scale])
        tadata_minmax = pd.DataFrame(scaled_data, columns=columns_to_scale)
        
        return tadata_minmax, columns_to_scale
        
    def get_relativeColumns(self) -> pd.DataFrame:
        """
        Retrieve selected technical indicators.

        Returns:
            pd.DataFrame: DataFrame containing the selected indicators.
        """
        # Check if all columns to keep are present
        missing_columns = [col for col in self.tacolumns_to_keep if col not in self.tadata.columns]
        if missing_columns:
            warnings.warn(f"The following columns are missing in tadata and will be skipped: {missing_columns}")
            # Use only the columns that are present
            columns_to_return = [col for col in self.tacolumns_to_keep if col in self.tadata.columns]
        else:
            columns_to_return = self.tacolumns_to_keep

        if not columns_to_return:
            raise ValueError("No columns available to return.")

        return self.tadata[columns_to_return], columns_to_return