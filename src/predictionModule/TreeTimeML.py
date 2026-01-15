import numpy as np
import polars as pl
import logging
import datetime

from sklearn.preprocessing import StandardScaler

from src.predictionModule.ModelAnalyzer import ModelAnalyzer
from src.mathTools.DistributionTools import DistributionTools
from src.predictionModule.LoadupSamples import LoadupSamples
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.predictionModule.WeightSamples import WeightSamples

from src.hyperparameterTuning.HelperMetrics import HelperMetrics
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

logger = logging.getLogger(__name__)

class TreeTimeML:
    # Class-level default parameters
    treetime_params = {
        "TreeTime_top_n": 5,
    }
    loadup_params = {
        "idxAfterPrediction": 5,

        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }
    precompute_params = {
        "FilterSamples_cat_over10": False,
        "FilterSamples_cat_under5000": False,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.2": False,
        "FilterSamples_cat_volatility_qdown0.02": False,
        "FilterSamples_cat_volatility_qup0.975": False,
        "FilterSamples_cat_predictability_qup0.9": False,
        "FilterSamples_cat_predictability_qdown0.1": False,
        
        "macd_diff_qup"                    : 0.929618,
        "macd_diff_qdown"                  : 0.998680,
        "macd_lookback_days"               : 2000, # trial.suggest_int("macd_lookback_days", 1500, 2500)
        "macd_fast_period"                 : 12,
        "macd_slow_period"                 : 56,
        "macd_signal_period"               : 8,
        "macd_fast_alpha"                  : 0.146235,
        "macd_slow_alpha"                  : 0.088318,
        "macd_signal_alpha"                : 0.121511,
    
        "atr_period"                       : 13,
        "atr_alpha"                        : 0.176763,
        "atr_lookback_days"                : 2000, # trial.suggest_int("atr_lookback_days", 1900, 2500)
        "atr_qup"                          : 0.922995,
        "atr_qdown"                        : 0.991166,
    }
    
    base_params = {
        "inc_FeatureTA"         : False, #trial.suggest_categorical("inc_FeatureTA", [True, False])
        "inc_GroupDynamics"     : False, #trial.suggest_categorical("inc_GroupDynamics", [True, False])
        "inc_Categorical"       : True, #trial.suggest_categorical("inc_Categorical", [True, False])
        "inc_Financials"        : True, #trial.suggest_categorical("inc_Financials", [True, False])
        "inc_Mathematical"      : False, #trial.suggest_categorical("inc_Mathematical", [True, False])
        "inc_Seasonal"          : True, #trial.suggest_categorical("inc_Seasonal", [True, False])
        "exc_lag"               : True, #trial.suggest_categorical("exc_lag", [True, False])
        
        "LGB_num_boost_round"           : 275, #trial.suggest_int("LGB_num_boost_round", 25, 300, step=25)
        "LGB_lambda_l1"                 : 0.000213, #trial.suggest_float("LGB_lambda_l1", 0.00001, 0.05, log=True)
        "LGB_lambda_l2"                 : 0.000124, #trial.suggest_float("LGB_lambda_l2", 0.0001, 0.02, log=True)
        "LGB_feature_fraction"          : 0.972680, #trial.suggest_float("LGB_feature_fraction", 0.94, 0.99, log=True)
        "LGB_num_leaves"                : 300, #trial.suggest_int("LGB_num_leaves", 300, 1200, step=25)
        "LGB_max_depth"                 : 9, #trial.suggest_int("LGB_max_depth", 4, 30, step=1)
        "LGB_learning_rate"             : 0.000511, #trial.suggest_float("LGB_learning_rate", 0.00001, 0.01, log=True)
        "LGB_min_data_in_leaf"          : 550, #trial.suggest_int("LGB_min_data_in_leaf", 100, 700, step=25)
        "LGB_min_gain_to_split"         : 0.000825, #trial.suggest_float("LGB_min_gain_to_split", 0.0001, 0.9, log=True)
        "LGB_path_smooth"               : 0.6, #trial.suggest_float("LGB_path_smooth", 0.01, 0.9, log=True)
        "LGB_min_sum_hessian_in_leaf"   : 0.2, #trial.suggest_float("LGB_min_sum_hessian_in_leaf", 0.001, 0.25, log=True)
        "LGB_max_bin"                   : 625, #trial.suggest_int("LGB_max_bin", 100, 800, step=25)
        "LGB_early_stopping_rounds"     : 27,
        
        "do_transform"      : False, #trial.suggest_categorical("do_transform", [True, False])
        "val_split"         : 0.008523, #trial.suggest_float("val_split", 0.001, 0.05, log=True)

        #opt_params["ytree_kind"] = trial.suggest_categorical("ytree_kind", ["last", "abslast"]) #mean and max not very good
        "ytr_opt_alpha"     : 0.014543, #trial.suggest_float("ytr_opt_alpha", 0.01, 0.2, log=True)
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
        
        self.params = {**self.loadup_params, **(params or {})}
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
                params=self.loadup_params,
            )
        else:
            ls = loadup
            if ls.treegroup != self.treegroup:
                raise ValueError("Provided LoadupSamples does not match the treegroup.")
            if ls.timegroup != self.timegroup:
                raise ValueError("Provided LoadupSamples does not match the timegroup.")
            if ls.train_start_date != self.train_start_date:
                raise ValueError("Provided LoadupSamples does not match the train start date.")
            if ls.test_dates != self.test_dates:
                raise ValueError("Provided LoadupSamples does not match the test dates.")
            if any(ls.params.get(k, None) is None for k, v in self.loadup_params.items()):
                raise ValueError("Provided LoadupSamples does not exist in the loadup parameters.")
            if any(ls.params.get(k) != v for k, v in self.loadup_params.items()):
                raise ValueError("Provided LoadupSamples does not match the loadup parameters.")
            self.test_dates = ls.test_dates
            
        self.idxDaysAfter = ls.idxAfter
        self.timesteps = ls.timesteps
        
        self.featureTreeNames: list[str] | None = ls.featureTreeNames
        self.featureTimeNames: list[str] | None = ls.featureTimeNames
        self.meta_pl_train: pl.DataFrame = ls.meta_pl_train
        self.meta_pl_test: pl.DataFrame  = ls.meta_pl_test
        
        self.train_Xtree: np.ndarray = ls.train_Xtree
        self.train_Xtime: np.ndarray = ls.train_Xtime
        self.train_ytree: np.ndarray = ls.train_ytree
        self.train_ytime: np.ndarray = ls.train_ytime
        
        self.test_Xtree: np.ndarray = ls.test_Xtree
        self.test_Xtime: np.ndarray = ls.test_Xtime
        self.test_ytree: np.ndarray = ls.test_ytree
        self.test_ytime: np.ndarray = ls.test_ytime
        
        self.train_ytree_low: np.ndarray  = ls.train_ytree_low
        self.train_ytree_open: np.ndarray = ls.train_ytree_open
        self.train_ytree_high: np.ndarray = ls.train_ytree_high
        self.test_ytree_low: np.ndarray   = ls.test_ytree_low
        self.test_ytree_open: np.ndarray  = ls.test_ytree_open
        self.test_ytree_high: np.ndarray  = ls.test_ytree_high
        
        self.mask_train = np.ones(self.train_Xtree.shape[0], dtype=bool)
        self.mask_test = np.ones(self.test_Xtree.shape[0], dtype=bool)
        
        self.sl_tr_vec = np.zeros(self.train_Xtree.shape[0], dtype=float)
        self.sl_te_vec = np.zeros(self.test_Xtree.shape[0], dtype=float)
        self.tp_tr_vec = np.zeros(self.train_Xtree.shape[0], dtype=float)
        self.tp_te_vec = np.zeros(self.test_Xtree.shape[0], dtype=float)

        
    def run_ext(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        ytr_tree_low: np.ndarray,
        ytr_tree_high: np.ndarray,
        ytr_tree_open: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame,
        meta_test: pl.DataFrame,
    ) -> tuple[np.ndarray, ...]:
        opt_params = self.base_params
        last_day_mask = (meta_test["date"] == meta_test["date"].max()).fill_null(False)
        logger.info(f"  All test dates {meta_test.get_column('date').unique().to_list()}")
        logger.info(f"  Selecting last day only: {meta_test["date"].max()}")
        
        def default_return():
            return (
                np.zeros(Xtr_tree.shape[0], dtype=bool), 
                np.zeros(Xte_tree.shape[0], dtype=bool), 
                0.88*np.ones(Xtr_tree.shape[0], dtype=float),
                0.88*np.ones(Xte_tree.shape[0], dtype=float),
                2.0*np.ones(Xtr_tree.shape[0], dtype=float), 
                2.0*np.ones(Xte_tree.shape[0], dtype=float),
                np.ones(Xtr_tree.shape[0], dtype=float), 
                np.ones(Xte_tree.shape[0], dtype=float),
            )
        
        if meta_test.is_empty() or meta_test['date'].is_empty() or (last_day_mask.to_numpy()).sum() <= 0:
            return default_return()
                
        mm: MachineModels = MachineModels(opt_params)
        do_transform =      opt_params["do_transform"]
        val_split =         opt_params["val_split"]
        
        tn = np.asarray(treenames, dtype=str)
        mask_treenames = np.zeros(len(treenames), dtype=bool)
        if opt_params.get("inc_FeatureTA"):
            mask_treenames |= np.char.find(tn, "FeatureTA_") >= 0
        if opt_params.get("inc_GroupDynamics"):
            mask_treenames |= np.char.find(tn, "FeatureGroup_") >= 0
        if opt_params.get("inc_Categorical"):
            mask_treenames |= np.char.find(tn, "Category_") >= 0
        if opt_params.get("inc_Financials"):
            mask_treenames |= np.char.find(tn, "FinData_") >= 0
        if opt_params.get("inc_Mathematical"):
            mask_treenames |= np.char.find(tn, "MathFeature_") >= 0
        if opt_params.get("inc_Seasonal"):
            mask_treenames |= np.char.find(tn, "Seasonal_") >= 0
        if opt_params.get("exc_lag"):
            mask_treenames &= np.char.find(tn, "_lag") < 0
            
        
        def to_time(x, inc_factor):
            return np.clip(np.tanh(np.log(np.clip(x, 1e-6, None)) * inc_factor) / 2.0 + 0.5, 1e-6, 1 - 1e-6)
        def to_tree(y, inc_factor):
            return np.exp(np.arctanh((y - 0.5) * 2.0)/inc_factor)    
        ytr_opt_alpha = opt_params.get("ytr_opt_alpha", 0.1)
        ytr_time_close = to_time(ytr_tree, 1.0)
        Xtr_time_close = Xtr_time[:, :, 0]
        close_comb = np.hstack((Xtr_time_close, ytr_time_close))
        ytr_opt = self._ema_2d(close_comb, alpha=ytr_opt_alpha)
        ytr_tree_opt = to_tree(ytr_opt[:, -1], 1.0)

        #if opt_params["ytree_kind"] == "last":
        #    ytr_tree_opt = ytr_tree[:, -1]
        #elif opt_params["ytree_kind"] == "mean":
        #    ytr_tree_opt = np.mean(ytr_tree, axis=1)
        #elif opt_params["ytree_kind"] == "abslast":
        #    ytr_tree_opt = np.abs(ytr_tree[:, -1])
        #elif opt_params["ytree_kind"] == "max":
        #    ytr_tree_opt = np.max(ytr_tree, axis=1)
            
        logger.info(f"  Before filtering: tr {Xtr_tree.shape}, te {Xte_tree.shape}")

        Xd_tr = Xtr_tree[:, mask_treenames]
        Xd_te = Xte_tree[:, mask_treenames]
        
        logger.info(f"  After design: tr {Xd_tr.shape}, te {Xd_te.shape}")
        logger.info(f"   ytr_tree: n={ytr_tree_opt.size}, mean={ytr_tree_opt.mean():.6f}, std={ytr_tree_opt.std():.6f}")

        if do_transform:
            scaler = StandardScaler().fit(Xd_tr)
            Xd_tr = scaler.transform(Xd_tr)
            Xd_te = scaler.transform(Xd_te)

        sam_split = int(Xd_tr.shape[0] * (1 - val_split))
        try:
            logger.disabled = True
            model_lgb, info = mm.run_LGB(
                X_train=Xd_tr[:sam_split],
                y_train=ytr_tree_opt[:sam_split],
                X_test=Xd_tr[sam_split:],
                y_test=ytr_tree_opt[sam_split:],
            )
        except Exception as e:
            logger.disabled = False
            logger.warning(f"  LGB failed: {e}")
            return 1.0
        finally:
            logger.disabled = False

        # LGB Predictions
        best_iter = getattr(model_lgb, "best_iteration", None)
        y_train_score = model_lgb.predict(Xd_tr, num_iteration=best_iter)
        y_test_score  = model_lgb.predict(Xd_te, num_iteration=best_iter)
        logger.info(f"  Test RMSE LGBM: {info['best_score']:.4f}")
                
        logger.info(f"  Run -> top scores test: {y_test_score} ")
        
        #sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
        #    ytr_tree, 
        #    ytr_tree_low, 
        #    ytr_tree_high, 
        #    ytr_tree_open
        #)
        sl_val, tp_val = 0.88, 2.0
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)
        
        mask_tr = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_te = np.ones(Xte_tree.shape[0], dtype=bool)

        return mask_tr, mask_te, sl_tr, sl_te, tp_tr, tp_te, y_train_score, y_test_score
    
        
    def precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        ytr_tree_low,
        ytr_tree_high,
        ytr_tree_open,
        Xte_tree,
        Xte_time,
        treenames,
        timenames,
        meta_train,
        meta_test,
    ) -> tuple[np.ndarray, np.ndarray]:
        if treenames is None or meta_train is None or meta_test is None:
            raise ValueError("treenames, meta_train and meta_test are required.")

        # --- MACD masks ---
        mask_train_macd, mask_test_macd = self._compute_macd_masks(
            Xtr_tree,
            Xte_tree,
            Xtr_time,
            Xte_time,
            meta_train,
            self.precompute_params,
        )

        # --- ATR masks ---
        mask_train_atr, mask_test_atr = self._compute_atr_masks(
            Xtr_time,
            Xte_time,
            meta_train,
            self.precompute_params,
        )

        # --- Extend MACD masks with ATR-only samples (union) ---
        mask_train = mask_train_macd | mask_train_atr
        mask_test = mask_test_macd | mask_test_atr
        
        logger.info(f"  Precompute masks: train kept {mask_train.sum()}/{mask_train.size}, test kept {mask_test.sum()}/{mask_test.size}")
        logger.info(f"  Precompute n masks: train {mask_train.sum()}, test {mask_test.sum()}")

        # If you later re-enable SL/TP optimization, use mask_train here:
        # sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
        #     ytr_tree[mask_train],
        #     ytr_tree_low[mask_train],
        #     ytr_tree_high[mask_train],
        #     ytr_tree_open[mask_train],
        #     n_grid=20,
        #     spread_cost=0.0,
        #     commission=0.0,
        # )

        sl_val, tp_val = 0.86, 2.0
        sl_tr_vec = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te_vec = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr_vec = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te_vec = tp_val * np.ones(Xte_tree.shape[0], dtype=float)

        return mask_train, mask_test, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_design(self, X, t_win):
        Xw: np.ndarray = X[:, -(t_win+1):, 0:5].copy()
        Xw = (Xw-0.5)*2.0
        
        mask_bad = np.zeros(Xw.shape[0], dtype=bool)
        bound_bad = 1 - np.tanh(1 - 1e-4)
        mask_bad = np.any((Xw[:,:,0:4] <= (-1+bound_bad)) | (Xw[:,:,0:4] >= (1-bound_bad)), axis=(1,2))
        Xw[:,:,0:4] = np.clip(Xw[:,:,0:4], -1+bound_bad, 1-bound_bad)
        
        Xw[:,:,0:4] = np.arctanh(Xw[:,:,0:4]) + 1.0
        
        return Xw.reshape(Xw.shape[0], -1), mask_bad

    def _ema_2d(
        self, 
        values: np.ndarray, 
        alpha: float
    ) -> np.ndarray:
        """
        values: (n_samples, n_steps)
        alpha: scalar EMA decay (0 < alpha <= 1)
        """
        out = np.empty_like(values, dtype=np.float64)
        out[:, 0] = values[:, 0]
        for t in range(1, values.shape[1]):
            out[:, t] = alpha * values[:, t] + (1.0 - alpha) * out[:, t - 1]
        return out

    def _true_range_2d(
        self,
        close: np.ndarray,  # (n_samples, n_steps)
        high: np.ndarray,   # (n_samples, n_steps)
        low: np.ndarray,    # (n_samples, n_steps)
    ) -> np.ndarray:
        """
        Vectorized True Range for each sample and time step.
        TR_t = max(
            high_t - low_t,
            |high_t - close_{t-1}|,
            |low_t  - close_{t-1}|
        ), with TR_0 = high_0 - low_0
        """
        close = close.astype(np.float64, copy=False)
        high = high.astype(np.float64, copy=False)
        low = low.astype(np.float64, copy=False)

        tr = np.empty_like(close, dtype=np.float64)
        # t = 0
        tr[:, 0] = high[:, 0] - low[:, 0]

        for t in range(1, close.shape[1]):
            prev_close = close[:, t - 1]
            tr1 = high[:, t] - low[:, t]
            tr2 = np.abs(high[:, t] - prev_close)
            tr3 = np.abs(low[:, t] - prev_close)
            tr[:, t] = np.maximum(tr1, np.maximum(tr2, tr3))

        return tr

    def _compute_macd_masks(
        self,
        Xtr_tree: np.ndarray,          # kept for interface compatibility, not used
        Xte_tree: np.ndarray,          # kept for interface compatibility, not used
        Xtr_time: np.ndarray,          # (n_samples, 90, 5)
        Xte_time: np.ndarray,          # (n_samples, 90, 5)
        meta_tr: pl.DataFrame,
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recomputes MACD and signal from time-series tensors and builds masks.
        Only X*_time[..., 0] (first feature) is used as the price series.
        """

        # --- MACD parameters from opt_params ---
        fast_period = opt_params["macd_fast_period"]
        slow_period = opt_params["macd_slow_period"]
        signal_period = opt_params["macd_signal_period"]
        max_period = max(fast_period, slow_period, signal_period)+1  # +1 for day-start adjustments in the computations

        # Per-line decay rates (if not provided, use standard EMA from periods)
        alpha_fast = opt_params.get("macd_fast_alpha", 2.0 / (fast_period + 1.0))
        alpha_slow = opt_params.get("macd_slow_alpha", 2.0 / (slow_period + 1.0))
        alpha_signal = opt_params.get("macd_signal_alpha", 2.0 / (signal_period + 1.0))

        # --- Extract price series (first feature) ---
        # shape: (n_samples, n_steps)
        close_tr = Xtr_time[:,-max_period:, 0].astype(np.float64, copy=False)
        close_te = Xte_time[:,-max_period:, 0].astype(np.float64, copy=False)

        # --- EMA fast/slow and MACD for train/test ---
        ema_fast_tr = self._ema_2d(close_tr, alpha_fast)
        ema_slow_tr = self._ema_2d(close_tr, alpha_slow)
        macd_tr_full = ema_fast_tr - ema_slow_tr

        ema_fast_te = self._ema_2d(close_te, alpha_fast)
        ema_slow_te = self._ema_2d(close_te, alpha_slow)
        macd_te_full = ema_fast_te - ema_slow_te

        # --- Signal line (EMA of MACD) ---
        sig_tr_full = self._ema_2d(macd_tr_full, alpha_signal)
        sig_te_full = self._ema_2d(macd_te_full, alpha_signal)

        # Use only the last time step as the feature value for each sample
        macd_tr_last = macd_tr_full[:, -1]
        sig_tr_last = sig_tr_full[:, -1]
        macd_te_last = macd_te_full[:, -1]
        sig_te_last = sig_te_full[:, -1]

        # --- Lookback in terms of samples (same logic as before) ---
        lookback_days = opt_params["macd_lookback_days"]
        n_days = meta_tr.get_column("date").n_unique()
        days_ratio = np.clip(lookback_days / n_days, 0.0, 1.0)

        n_samples_lookback = int(np.ceil(days_ratio * Xtr_time.shape[0]))
        n_samples_lookback = int(np.clip(n_samples_lookback, 1, Xtr_time.shape[0]))

        # Quantiles are computed from the last n_samples_lookback train samples
        diff_tr_lookback = macd_tr_last[-n_samples_lookback:] - sig_tr_last[-n_samples_lookback:]

        qdown = opt_params["macd_diff_qdown"]
        qup = opt_params["macd_diff_qup"]
        qup_val = np.quantile(diff_tr_lookback, qup)
        qdown_val = np.quantile(diff_tr_lookback, qdown)

        # --- Apply thresholds to all samples ---
        diff_tr_all = macd_tr_last - sig_tr_last
        diff_te_all = macd_te_last - sig_te_last

        # NOTE: It is that qdown > qup , hence [qup_val, qdown_val] interval
        mask_tr = (diff_tr_all <= qdown_val) & (diff_tr_all >= qup_val)
        mask_te = (diff_te_all <= qdown_val) & (diff_te_all >= qup_val)

        return mask_tr, mask_te
    
    def _compute_atr_masks(
        self,
        Xtr_time: np.ndarray,      # (n_samples, n_steps, n_features)
        Xte_time: np.ndarray,      # (n_samples, n_steps, n_features)
        meta_tr: pl.DataFrame,
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Builds train/test masks based on Average True Range (ATR).
        Uses features:
            0: AdjClose
            2: AdjHigh
            3: AdjLow
        Threshold is a quantile of ATR over a lookback on train.
        """

        # --- ATR parameters ---
        atr_period = opt_params["atr_period"]
        atr_alpha = opt_params.get("atr_alpha", 2.0 / (atr_period + 1.0))

        # Re-use MACD lookback unless an ATR-specific one is provided
        lookback_days = opt_params.get("atr_lookback_days")
        atr_qup = opt_params["atr_qup"]  # e.g. 0.7 for top 30% ATR
        atr_qdown = opt_params["atr_qdown"]  # e.g. 0.0 for no lower bound

        # --- Extract close / high / low ---
        close_tr = Xtr_time[:, -(atr_period+1):, 0].astype(np.float64, copy=False)
        high_tr = Xtr_time[:, -(atr_period+1):, 2].astype(np.float64, copy=False)
        low_tr  = Xtr_time[:, -(atr_period+1):, 3].astype(np.float64, copy=False)

        close_te = Xte_time[:, -(atr_period+1):, 0].astype(np.float64, copy=False)
        high_te = Xte_time[:, -(atr_period+1):, 2].astype(np.float64, copy=False)
        low_te  = Xte_time[:, -(atr_period+1):, 3].astype(np.float64, copy=False)

        # --- True Range and ATR (EMA of TR) ---
        tr_tr = self._true_range_2d(close_tr, high_tr, low_tr)
        tr_te = self._true_range_2d(close_te, high_te, low_te)

        atr_tr_full = self._ema_2d(tr_tr, atr_alpha)
        atr_te_full = self._ema_2d(tr_te, atr_alpha)

        # Use last time step per sample
        atr_tr_last = atr_tr_full[:, -1]
        atr_te_last = atr_te_full[:, -1]

        # --- Lookback in terms of samples (same logic as MACD) ---
        n_days = meta_tr.get_column("date").n_unique()
        days_ratio = np.clip(lookback_days / n_days, 0.0, 1.0)

        n_samples_lookback = int(np.ceil(days_ratio * Xtr_time.shape[0]))
        n_samples_lookback = int(np.clip(n_samples_lookback, 1, Xtr_time.shape[0]))

        atr_tr_lookback = atr_tr_last[-n_samples_lookback:]
        atr_qup = np.quantile(atr_tr_lookback, atr_qup)
        atr_qdown = np.quantile(atr_tr_lookback, atr_qdown)

        # High-ATR regime
        mask_tr = (atr_tr_last >= atr_qup) & (atr_tr_last <= atr_qdown)
        mask_te = (atr_te_last >= atr_qup) & (atr_te_last <= atr_qdown)

        return mask_tr, mask_te    
    
    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    
    def _pipeline_res_analyze(self, stage: str, mode: str = "predict") -> None:
        res_tr_vec = HelperMetrics.collapse_sl_tp(
            self.train_ytree, 
            self.train_ytree_low, 
            self.train_ytree_high, 
            self.train_ytree_open, 
            self.sl_tr_vec, 
            self.tp_tr_vec, 
            spread_cost=0.000, 
            commission=0.000
        )
        if mode == "analyze":
            res_te_vec = HelperMetrics.collapse_sl_tp(
                    self.test_ytree, 
                    self.test_ytree_low, 
                    self.test_ytree_high, 
                    self.test_ytree_open, 
                    self.sl_te_vec, 
                    self.tp_te_vec, 
                    spread_cost=0.000, 
                    commission=0.000
                )
        logger.info(f"  After {stage}: ")
        logger.info(f"    Ratio training samples: {self.mask_train.sum() / len(self.mask_train)}")
        logger.info(f"    Ratio test samples: {self.mask_test.sum() / len(self.mask_test)}")
        
        pre_tr_score = HelperMetrics.evaluate_mask_nullonempty(self.mask_train, self.meta_pl_train['date'], res_tr_vec)
        logger.info(f"    {stage} train score: {pre_tr_score:.4f}")
        
        if mode == "analyze":
            pre_te_score = HelperMetrics.evaluate_mask_nullonempty(self.mask_test, self.meta_pl_test['date'], res_te_vec)
            logger.info(f"    {stage} test score: {pre_te_score:.4f}")
    
            sl_hits = (self.test_ytree_low[self.mask_test][:, -1] <= self.sl_te_vec[self.mask_test])
            tp_hits = (self.test_ytree_high[self.mask_test][:, -1] >= self.tp_te_vec[self.mask_test])
            tp_nosl_hits = tp_hits & ~sl_hits
            logger.info(f"    {stage} ratio sl hits test split: {np.sum(sl_hits)/len(self.sl_te_vec[self.mask_test]):.4f}, n_test_samples = {len(self.sl_te_vec[self.mask_test])}")
            logger.info(f"    {stage} ratio tp hits test split: {np.sum(tp_hits)/len(self.tp_te_vec[self.mask_test]):.4f}, n_test_samples = {len(self.tp_te_vec[self.mask_test])}")
            logger.info(f"    {stage} ratio tp no sl hits test split: {np.sum(tp_nosl_hits)/len(self.tp_te_vec[self.mask_test]):.4f}, n_test_samples = {len(self.tp_te_vec[self.mask_test])}")
        
    def pipeline(self, mode: str = "predict") -> dict:
        """
        Common pipeline steps shared by both analyze() and predict().
        Returns a dictionary of all relevant masked data, trained model, and predictions.
        """

        ########################
        ## PRE FILTER SAMPLES ##
        ########################
        logger.info("Running pre-filtering samples...")
        try:
            mask_tr, mask_te, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec = self.precompute(
                self.train_Xtree.copy(),
                self.train_Xtime.copy(),
                self.train_ytree.copy(),
                self.train_ytree_low.copy(),
                self.train_ytree_high.copy(),
                self.train_ytree_open.copy(),
                self.test_Xtree.copy(),
                self.test_Xtime.copy(),
                self.featureTreeNames,
                self.featureTimeNames,
                self.meta_pl_train,
                self.meta_pl_test,
            )
        except Exception as e:
            logger.warning(f"  Error occurred while pre-filtering samples: {e}")
            return {
                'y_test_scores': np.zeros(self.test_Xtree.shape[0], dtype=float),
                'mask_train': np.zeros(self.train_Xtree.shape[0], dtype=bool),
                'mask_test': np.zeros(self.test_Xtree.shape[0], dtype=bool),
            }
        self.sl_tr_vec = sl_tr_vec
        self.sl_te_vec = sl_te_vec
        self.tp_tr_vec = tp_tr_vec
        self.tp_te_vec = tp_te_vec
        self.mask_train = mask_tr
        self.mask_test = mask_te
        
        self._pipeline_res_analyze(
            stage="Pre-filter", 
            mode=mode, 
        )


        #########################
        ## MAIN FILTER SAMPLES ##
        #########################
        logger.info("Running main filtering samples...")
        try:
            mask_tr, mask_te, sl_tr_vec, sl_te_vec, tp_tr_vec, tp_te_vec, vecscore_tr_vec, vecscore_te_vec = self.run_ext(
                self.train_Xtree[self.mask_train],
                self.train_Xtime[self.mask_train],
                self.train_ytree[self.mask_train],
                self.train_ytree_low[self.mask_train],
                self.train_ytree_high[self.mask_train],
                self.train_ytree_open[self.mask_train],
                self.test_Xtree[self.mask_test],
                self.test_Xtime[self.mask_test],
                self.featureTreeNames,
                self.featureTimeNames,
                self.meta_pl_train.filter(pl.Series(self.mask_train)),
                self.meta_pl_test.filter(pl.Series(self.mask_test)),
            )
        except Exception as e:
            logger.warning(f"  Error occurred while main-filtering samples: {e}")
            return {
                'y_test_scores': np.zeros(self.test_Xtree.shape[0], dtype=float),
                'mask_train': np.zeros(self.train_Xtree.shape[0], dtype=bool),
                'mask_test': np.zeros(self.test_Xtree.shape[0], dtype=bool),
            }
        self.sl_tr_vec[self.mask_train]  = sl_tr_vec
        self.sl_te_vec[self.mask_test]   = sl_te_vec
        self.tp_tr_vec[self.mask_train]  = tp_tr_vec
        self.tp_te_vec[self.mask_test]   = tp_te_vec
        
        self.mask_train[self.mask_train] = mask_tr
        self.mask_test[self.mask_test]   = mask_te
        
        self._pipeline_res_analyze(
            stage="Main-run", 
            mode=mode, 
        )

        score_te_full = np.zeros(self.test_Xtree.shape[0], dtype=float)
        score_te_full[self.mask_test] = vecscore_te_vec[mask_te]

        #############
        ## RETURNS ##
        #############
        return {
            'y_test_scores': score_te_full,
            'mask_train': self.mask_train,
            'mask_test': self.mask_test,
        }

    def __get_top_tickers(self, y_test_scores: np.ndarray) -> pl.DataFrame:
        m = self.treetime_params['TreeTime_top_n']
        
        res_pl = self.meta_pl_test.filter(pl.Series(self.mask_test)).with_columns(
            pl.Series("test_scores", y_test_scores[self.mask_test]),
        )
        
        res_pl = (
            res_pl
            .sort(["date", "test_scores"], descending=[False, True])
            .with_columns(
                pl.col("test_scores")
                .rank(method="random", descending=True)
                .over("date")
                .alias("score_rank")
            ).filter(
                pl.col("score_rank") <= m
            )
        )

        return res_pl
    
    def _get_res_df(self, y_test_scores: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame]:
        m = self.treetime_params['TreeTime_top_n']
        
        collapsed_te_vec = HelperMetrics.collapse_sl_tp(
            self.test_ytree[self.mask_test], 
            self.test_ytree_low[self.mask_test], 
            self.test_ytree_high[self.mask_test], 
            self.test_ytree_open[self.mask_test], 
            self.sl_te_vec[self.mask_test], 
            self.tp_te_vec[self.mask_test], 
            spread_cost=0.000, 
            commission=0.000
        )
        
        res_pl = self.meta_pl_test.filter(pl.Series(self.mask_test)).with_columns(
            pl.Series("collapsed_ratio", collapsed_te_vec),
            pl.Series("test_scores", y_test_scores[self.mask_test]),
        )
        
        res_pl = (
            res_pl
            .sort(["date", "test_scores"], descending=[False, True])
            .with_columns(
                pl.col("test_scores")
                .rank(method="random", descending=True)
                .over("date")
                .alias("test_scores_rank")
            ).filter(
                pl.col("test_scores_rank") <= m
            )
        )
        
        res_pl_perdate = (
            res_pl.group_by("date").agg([
                pl.count().alias("n_entries"),
                pl.col("collapsed_ratio").mean().alias("mean_collapsed_ratio"),
            ])
        )
        
        return res_pl, res_pl_perdate
    
    def analyze(self, logger_disabled: bool = False) -> tuple[float, dict]:
        logger_config = logger.disabled
        logger.disabled = logger_disabled
        
        # Run common pipeline in "analyze" mode
        data = self.pipeline(mode="analyze")
        
        # Additional analysis with test set
        y_test_scores: np.ndarray = data['y_test_scores']
        mask_test: np.ndarray = data['mask_test']
        
        if mask_test.sum() == 0:
            return (
                1.0, 
                {
                    "res_df": pl.DataFrame(),
                    "res_df_perdate": pl.DataFrame(),
                }
            )
        
        res_df, res_df_perdate = self._get_res_df(y_test_scores)
        
        logger.info("Analyzing test set predictions:")
        logger.info(f"  Number of test dates: {len(self.test_dates)}")
        logger.info(f"  Ratio of test dates with choices: {res_df_perdate.shape[0] / len(self.test_dates):.4f}")

        score_col = "test_scores"
        tar_col = "collapsed_ratio"
        ModelAnalyzer.log_test_result_perdate(res_df, self.test_dates, score_col = score_col, tar_col = tar_col)
        ModelAnalyzer.log_test_result_overall(res_df, score_col = score_col, last_col = tar_col)

        res_df_perdate = res_df.group_by("date").agg([
            pl.col(tar_col).mean().alias("mean_res"),
            pl.col(tar_col).first().alias("top_res"),
            pl.col(tar_col).count().alias("n_entries"),
            pl.col(score_col).max().alias("max_score"),  # this is also .first()
            pl.col(score_col).mean().alias("mean_score"),
        ])

        logger.disabled = logger_config
        return (
            res_df_perdate['mean_score'].mean(), 
            {
                "res_df": res_df.select(['date', 'ticker', 'Close', score_col, tar_col]),
                "res_df_perdate": res_df_perdate.select(['date', 'n_entries', 'max_score', 'mean_score', 'top_res', 'mean_res'])
            }
        )

    def predict(self, logger_disabled: bool = False) -> tuple[float, dict]:
        logger_config = logger.disabled
        logger.disabled = logger_disabled
        
        # Run common pipeline in "analyze" mode
        data = self.pipeline(mode="predict")

        # Additional analysis with test set
        y_test_scores: np.ndarray = data['y_test_scores']
        mask_test: np.ndarray = data['mask_test']
        
        if mask_test.sum() == 0:
            return (
                1.0, 
                {
                    "res_df": pl.DataFrame(),
                    "res_df_perdate": pl.DataFrame(),
                }
            )

        res_df = self.__get_top_tickers(y_test_scores)

        score_col = "test_scores"
        ModelAnalyzer.log_test_result_perdate(res_df, self.test_dates, score_col = score_col, tar_col = None)
        ModelAnalyzer.log_test_result_overall(res_df, score_col = score_col, last_col = None)

        res_df_perdate = res_df.group_by("date").agg([
            pl.col(score_col).count().alias("n_entries"),
            pl.col(score_col).mean().alias("mean_score"),
            pl.col(score_col).max().alias("max_score"),
        ])

        logger.disabled = logger_config
        return (
            res_df_perdate['mean_score'].mean(), 
            {
                "res_df": res_df.select(['date', 'ticker', 'Close', score_col]),
                "res_df_perdate": res_df_perdate.select(['date', 'n_entries', 'max_score', 'mean_score']),
            }
        )