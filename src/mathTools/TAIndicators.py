import numpy as np
import pandas as pd
import polars as pl
from ta import add_all_ta_features

class TAIndicators():
    tacolumns_SpecialPreprocessed = [
        'volume_em',                    # Ease of Movement: divide by 100 then sigmoid. needs more testing
        'volume_sma_em',                # ?: divide by 500 then sigmoid. needs more testing
        'volume_nvi',                   # Negative Volume Index: divide by 100 then log
        'volatility_kcp',               # Keltner Channel %: clip -10 10 then divide by 10
        'trend_dpo',                    # Detrended Price Oscillator: divide by 100 then sigmoid
        'trend_kst',                    # Know Sure Thing: divide by 100 then sigmoid
        'trend_kst_sig',                # KST Signal: divide by 100 then sigmoid
        'trend_kst_diff',               # KST Difference: divide by 100 then sigmoid
        'trend_adx',                    # Average Directional Movement Index: divide by 100 then sigmoid
        'trend_adx_pos',                # ADX Positive: divide by 100 then sigmoid
        'trend_adx_neg',                # ADX Negative: divide by 100 then sigmoid
        'momentum_ppo',                 # Percentage Price Oscillator: divide by 10, then sigmoid
        'momentum_ppo_signal',          # Percentage Price Oscillator Signal: divide by 10, then sigmoid
        'momentum_ppo_hist',            # Percentage Price Oscillator Histogram: divide by 10, then sigmoid   
        'volume_obv',                   # On-balance Volume: divide by its rolling max. Lag imporant. Ideal: Ratio to NumberOfOutstandingShares
        'volume_fi',                    # Force Index: Descale by its rolling max
        'volume_vpt',                   # Volume-price Trend: divide by its rolling max
        'trend_trix',                   # Trix: zero out first 5 values, then divide by 10 then sigmoid 
    ]
    tacolumns_DivideByHundreds = [
        'volume_mfi',                   # Money Flow Index: divide by 100. 
        'volatility_ui',                # Ulcer Index: divide by 100
        'trend_mass_index',             # Mass Index: divide by 100
        'trend_aroon_up',               # Aroon Up: divide by 100
        'trend_aroon_down',             # Aroon Down: divide by 100
        'trend_aroon_ind',              # Aroon Indicator: divide by 100
        'momentum_tsi',                 # True Strength Index: divide by 100
        'momentum_uo',                  # Ultimate Oscillator: divide by 100
        'momentum_stoch',               # Stochastic Oscillator: divide by 100
        'momentum_stoch_signal',        # Stochastic Signal: divide by 100
        'momentum_wr',                  # Williams %R: divide by 100
        'momentum_roc',                 # Rate of Change: divide by 100
        'trend_stc',                    # Schaff Trend Cycle: divide by 100 
        'trend_cci',                    # Commodity Channel Index: divide by 100
        'momentum_pvo_signal',          # Percentage Volume Oscillator Signal: divide by 100
        'momentum_rsi',                 # Relative Strength Index: divide by 100
        'momentum_pvo',                 # Percentage Volume Oscillator: divide by 100
        'momentum_pvo_hist',            # Percentage Volume Oscillator Histogram: divide by 100
    ]
    tacolumns_AsIs = [
        'volume_cmf',                   # Chaikin Money Flow: Is between -1 and 1. Do not scale
        'volatility_bbhi',              # Bollinger Bands %B High: binary (or ternary), leave as is
        'trend_psar_down_indicator',    # Parabolic SAR Down Indicator: binary (or ternary), leave as is
        'volatility_kcli',              # Keltner Channel Low ind: binary (or ternary), leave as is
        'volatility_bbli',              # Bollinger Bands Low ind: binary (or ternary), leave as is
        'trend_psar_up_indicator',      # Parabolic SAR Up Indicator: binary (or ternary), leave as is
        'volatility_kchi',              # Keltner Channel High ind: binary (or ternary), leave as is
        'momentum_stoch_rsi',           # Stoch RSI: leave as is
        'momentum_stoch_rsi_k',         # Stoch RSI K: leave as is
        'momentum_stoch_rsi_d',         # Stoch RSI D: leave as is
        'volatility_dcp',               # Donchian Channel %: Leave as is
        'trend_vortex_ind_neg',         # Vortex Indicator Neg: leave it be
        'trend_vortex_ind_pos',         # Vortex Indicator Pos: leave it be
        'trend_vortex_ind_diff',        # Vortex Indicator: leave it be
        'volatility_bbp',               # Bollinger Bands %B: LEAVE AS IS. Lag important
    ]
    tacolumns_ScaledSpecial = [
        'Volume',                       # Volume: divide by cur Volume (not series)
        'volume_adi',                   # Accumulation Distribution Index: deScale to cur Volume*price 
    ]
    tacolumns_ScaledToClose = [
        'Open',
        'High',
        'Low',
        'trend_macd',               # MACD: divide by cur price (not series)
        'trend_macd_signal',        # MACD Signal
        'trend_macd_diff',          # MACD Difference
        'trend_sma_slow',           # Simple Moving Average Slow 
        'volatility_atr',           # Average True Range
        'volatility_bbl',           # Bollinger Bands Low
        'volatility_bbm',           # Bollinger Bands Mid
        'volatility_bbh',           # Bollinger Bands High
        'volatility_bbw',           # Bollinger Bands Width
        'volatility_kcc',           # Keltner Channel Center
        'momentum_ao',              # Awesome Oscillator
        'trend_psar_down',          # Parabolic SAR Down
        'trend_psar_up',            # Parabolic SAR Up
        'trend_sma_fast',           # Simple Moving Average Fast
        'volatility_kch',           # Keltner Channel High
        'volatility_kcl',           # Keltner Channel Low
        'volume_vwap',              # Volume Weighted Average Price
        'trend_ichimoku_base',      # Ichimoku Base Line
        'trend_ichimoku_conv',      # Ichimoku Conversion Line
        'trend_ichimoku_a',         # Ichimoku A
        'trend_visual_ichimoku_a',  # Ichimoku A
        'trend_visual_ichimoku_b',  # Ichimoku B
        'trend_ichimoku_b',         # Ichimoku B
        'trend_ema_slow',           # Exponential Moving Average Slow
        'momentum_kama',            # Kaufman's Adaptive Moving Average
        'trend_ema_fast',           # Exponential Moving Average Fast
        'volatility_dcm',           # Donchian Channel Mid
        'volatility_dcl',           # Donchian Channel Low
        'volatility_dch',           # Donchian Channel High
    ]
    
    notToUse = [
        'Close',
        'volatility_kcw', # Keltner Channel Width: we use the % instead
        'volatility_dcw', # Donchian Channel Width: we use the % instead
        'others_dlr',     # Daily Log Return: already used in MathFeature
        'others_dr',      # Daily Return: already used in MathFeature
        'others_cr',      # Cumulative Return: already used in MathFeature
    ]
    
    tacolumns_selectionTimeseries = [
        #special
        'volume_nvi',
        'Volume',
        'momentum_roc',
        'trend_aroon_ind',
        'momentum_pvo',
        'momentum_stoch_rsi',
        'momentum_stoch_rsi_k',
        'momentum_stoch_rsi_d',
        'volume_cmf',
        
        # to close
        'High',
        'Low',
        'trend_macd',
        'trend_macd_signal',
        'trend_sma_slow',
        'trend_ema_fast',
        'trend_ema_slow',
        'volatility_atr',
        'volatility_bbm',
        'momentum_ao',
        'trend_visual_ichimoku_b',
        
        # already sigmoided
        'trend_adx',
        'trend_kst',
        'trend_kst_sig',
        'trend_kst_diff',
        'momentum_ppo',
        
        # rolling maxed
        'volume_obv',
        'volume_vpt',
        
        #to clip
        'trend_mass_index',
        'trend_stc',
        'volatility_ui',
        'momentum_stoch',
    ]
        
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
            raise TypeError("Input DataFrame must be a pandas DataFrame or polars DataFrame.")

        # Polars is faster still
        self.tadata: pl.DataFrame = pl.from_pandas(self.tadata)
        self.tadata_columns = self.tadata.columns
        
        assert all([col in self.tadata_columns for col in self.tacolumns_ScaledSpecial]) , f"Missing columns: {set(self.tacolumns_ScaledSpecial) - set(self.tadata_columns)}"
        assert all([col in self.tadata_columns for col in self.tacolumns_SpecialPreprocessed]) , f"Missing columns: {set(self.tacolumns_SpecialPreprocessed) - set(self.tadata_columns)}"
        assert all([col in self.tadata_columns for col in self.tacolumns_DivideByHundreds]) , f"Missing columns: {set(self.tacolumns_DivideByHundreds) - set(self.tadata_columns)}"
        assert all([col in self.tadata_columns for col in self.tacolumns_AsIs]) , f"Missing columns: {set(self.tacolumns_AsIs) - set(self.tadata_columns)}"
        assert all([col in self.tadata_columns for col in self.tacolumns_ScaledToClose]) , f"Missing columns: {set(self.tacolumns_ScaledToClose) - set(self.tadata_columns)}"
        
        self.rollingbuffer = 21*12  # for calculating the rolling max
        
        all_ta_columns = (
            self.tacolumns_ScaledToClose + 
            self.tacolumns_ScaledSpecial + 
            self.tacolumns_SpecialPreprocessed + 
            self.tacolumns_DivideByHundreds + 
            self.tacolumns_AsIs
        )
        assert len(all_ta_columns) == len(set(all_ta_columns)), "Duplicate columns in TA columns"
        
        self.__preprocess()
        
    def __preprocess(self):
        """
        Preprocess the technical indicators by scaling and transforming them.
        """
        sigmoid_100_scale = [
            "volume_em",
            "trend_dpo",
            "trend_kst",
            "trend_kst_sig",
            "trend_kst_diff",
            "trend_adx",
            "trend_adx_pos",
            "trend_adx_neg",
        ]
        sigmoid_10_scale = [
            "momentum_ppo",
            "momentum_ppo_signal",
            "momentum_ppo_hist",
        ]
        
        def sigmoid_expression(col: pl.Expr, scale=1) -> pl.Expr:
            return 1 / (1 + (-col / scale).exp())

        self.tadata = self.tadata.with_row_count("row_idx")
        
        #Preprocess tacolumns_SpecialPreprocessed
        self.tadata = self.tadata.with_columns(
            [sigmoid_expression(pl.col(col), scale=100.0).alias(col) for col in sigmoid_100_scale]
        )
        self.tadata = self.tadata.with_columns([
            sigmoid_expression(pl.col("volume_sma_em"), scale=500.0).alias("volume_sma_em"),
            (pl.col("volume_nvi") / 100).log().alias("volume_nvi"),
            (pl.col("volatility_kcp").clip(-10, 10) / 10).alias("volatility_kcp")
        ])
        self.tadata = self.tadata.with_columns(
            [sigmoid_expression(pl.col(col), scale=10.0).alias(col) for col in sigmoid_10_scale]
        )
        self.tadata = self.tadata.with_columns([
            (pl.col("volume_obv") / pl.col("volume_obv").rolling_max(self.rollingbuffer)).alias("volume_obv"),
            (pl.col("volume_fi") / pl.col("volume_fi").rolling_max(self.rollingbuffer)).alias("volume_fi"),
            (pl.col("volume_vpt") / pl.col("volume_vpt").rolling_max(self.rollingbuffer)).alias("volume_vpt"),
            #Preprocessing trend_trix
            pl.when(pl.col("row_idx") < 5).then(0).otherwise(pl.col("trend_trix")).alias("trend_trix"),
        ])
        self.tadata = self.tadata.with_columns([
            sigmoid_expression(pl.col("trend_trix"), scale=2).alias("trend_trix")
        ])    
        
        #Preprocess tacolumns_DivideByHundreds
        self.tadata = self.tadata.with_columns(
            [(pl.col(col)/ 100).alias(col) for col in self.tacolumns_DivideByHundreds]
        )
        
    def getTAColumnNames(self) -> list[str]:
        return (
            self.tacolumns_ScaledToClose + 
            self.tacolumns_ScaledSpecial + 
            self.tacolumns_SpecialPreprocessed + 
            self.tacolumns_DivideByHundreds + 
            self.tacolumns_AsIs
        )
        
    def getTAColumnNames_timeseries(self) -> list[str]:
        return self.tacolumns_selectionTimeseries
        
    def getReScaledDataFrame(self, curClosePrice: float, curVolume: float) -> pl.DataFrame:
        colNames = self.getTAColumnNames()
        
        # Scale the TA indicators based on the current close price and volume
        tadata_rescaled = self.tadata.select(colNames)
        
        # tacolumns_ScaledToClose
        tadata_rescaled = tadata_rescaled.with_columns(
            [(pl.col(col) / curClosePrice).alias(col) for col in self.tacolumns_ScaledToClose]
        )
        
        # tacolumns_ScaledSpecial
        tadata_rescaled = tadata_rescaled.with_columns([
            (pl.col('Volume') / curVolume).alias('Volume'),
            (pl.col("volume_adi") / (curVolume * curClosePrice)).alias("volume_adi"),
        ])
        
        return tadata_rescaled
    
    def getReScaledDataFrame_timeseries(self, curClosePrice: float, curVolume: float) -> pl.DataFrame:
        
        def tanh_R(col: pl.Expr) -> pl.Expr:
            return col.tanh()/2.0 + 0.5
        def tanh_Rplus(col: pl.Expr, scale=1.0) -> pl.Expr:
            return (col/scale).tanh()/2.0 + 0.5
        def tanh_R_centerOne(col: pl.Expr) -> pl.Expr:
            return (col-1.0).tanh()/2.0 + 0.5
        def clipExpr(col: pl.Expr) -> pl.Expr:
            return col.clip(0, 1)
        def lin_m1to1(col: pl.Expr) -> pl.Expr:
            return ((col + 1.0) / 2.0).clip(0, 1)
        
        colNamesTs = self.getTAColumnNames_timeseries()
        
        tadata_rescaled = self.tadata.select(colNamesTs)
        
        # tacolumns_ScaledToClose
        tadata_rescaled = tadata_rescaled.with_columns([
            tanh_R_centerOne(pl.col("High") / curClosePrice).alias("High"),
            tanh_R_centerOne(pl.col("Low") / curClosePrice).alias("Low"),
            tanh_R_centerOne(pl.col("trend_macd") / curClosePrice).alias("trend_macd"),
            tanh_R_centerOne(pl.col("trend_macd_signal") / curClosePrice).alias("trend_macd_signal"),
            tanh_R_centerOne(pl.col("trend_sma_slow") / curClosePrice).alias("trend_sma_slow"), 
            tanh_R_centerOne(pl.col("trend_ema_fast") / curClosePrice).alias("trend_ema_fast"),
            tanh_R_centerOne(pl.col("trend_ema_slow") / curClosePrice).alias("trend_ema_slow"),
            tanh_R_centerOne(pl.col("volatility_atr") / curClosePrice).alias("volatility_atr"),
            tanh_R_centerOne(pl.col("volatility_bbm") / curClosePrice).alias("volatility_bbm"),
            tanh_R_centerOne(pl.col("momentum_ao") / curClosePrice).alias("momentum_ao"),
            tanh_R_centerOne(pl.col("trend_visual_ichimoku_b") / curClosePrice).alias("trend_visual_ichimoku_b"),
        ])
        
        #special 
        tadata_rescaled = tadata_rescaled.with_columns([
            tanh_R(pl.col('volume_nvi').diff(n=21).fill_null(0)).alias('volume_nvi'),  #TODO: MH instead of diff would be better
            tanh_R_centerOne(pl.col('Volume') / (curVolume + 1e-8)).alias('Volume'),
            lin_m1to1(pl.col('momentum_roc')).alias('momentum_roc'),
            lin_m1to1(pl.col('trend_aroon_ind')).alias('trend_aroon_ind'),
            lin_m1to1(pl.col('momentum_pvo')).alias('momentum_pvo'),
            clipExpr(pl.col('momentum_stoch_rsi')).alias('momentum_stoch_rsi'),
            clipExpr(pl.col('momentum_stoch_rsi_k')).alias('momentum_stoch_rsi_k'),
            clipExpr(pl.col('momentum_stoch_rsi_d')).alias('momentum_stoch_rsi_d'),
            lin_m1to1(pl.col('volume_cmf')).alias('volume_cmf'),
        ])
        
        # sigmoided
        tadata_rescaled = tadata_rescaled.with_columns([
            clipExpr(pl.col('trend_adx')).alias('trend_adx'),
            clipExpr(pl.col('trend_kst')).alias('trend_kst'),
            clipExpr(pl.col('trend_kst_sig')).alias('trend_kst_sig'),
            clipExpr(pl.col('trend_kst_diff')).alias('trend_kst_diff'),
            clipExpr(pl.col('momentum_ppo')).alias('momentum_ppo'),
        ])
        
        # rolling maxed
        tadata_rescaled = tadata_rescaled.with_columns([
            clipExpr(pl.col('volume_obv')).alias('volume_obv'),
            tanh_Rplus(pl.col('volume_vpt'), 100.0).alias('volume_vpt'),
        ])
        
        # to clip
        tadata_rescaled = tadata_rescaled.with_columns([
            clipExpr(pl.col("trend_mass_index")).alias("trend_mass_index"),
            clipExpr(pl.col("trend_stc")).alias("trend_stc"),
            clipExpr(pl.col("volatility_ui")).alias("volatility_ui"),
            clipExpr(pl.col("momentum_stoch")).alias("momentum_stoch"),
        ])
        
        return tadata_rescaled