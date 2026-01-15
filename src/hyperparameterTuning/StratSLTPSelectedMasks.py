import numpy as np
import optuna
import polars as pl

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.hyperparameterTuning.HelperFunctions import HelperFunctions
from src.hyperparameterTuning.HelperSLTP import HelperSLTP

from src.hyperparameterTuning.StratSelectedMasks import StratSelectedMasks

import logging
logger = logging.getLogger(__name__)

class StratSLTPSelectedMasks(BaseStrategy):
    # Note Xtr_time needs the features
    #        "FeatureLSTM_AdjClose" at index 0,
    #        "FeatureLSTM_AdjOpen" at index 1,
    #        "FeatureLSTM_AdjHigh" at index 2,
    #        "FeatureLSTM_AdjLow" at index 3,
    
    expected_load_params = {
        "idxAfterPrediction": 5,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "atr_period"        : 5,   
        "atr_alpha"         : 0.171537,    
        "atr_qup"           : 0.848168,      
        "atr_qdown"         : 0.985,    
        "slope_period"      : 8,  
        "slope_qup"         : 0.632790,    
        "slope_qdown"       : 1.0, 
        "rmse_period"       : 32,  
        "rmse_delay"        : 4,   
        "rmse_ma_wndw"      : 17, 
        "rmse_alpha"        : 0.145176,   
        "rmse_qup"          : 0.732962,     
        "rmse_qdown"        : 0.987,   
        
        "sl0" : 0.823295, 
        "sl1" : 0.833562, 
        "sl2" : 0.783671, 
        "sl3" : 0.878303, 
        "sl4" : 0.982187, 
    
        "tp0" : 1.307256, 
        "tp1" : 1.442779, 
        "tp2" : 1.377516, 
        "tp3" : 1.211979, 
        "tp4" : 1.378628, 
    }

    base_params = {
    }

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        opt_params = {}
        opt_params["min_n_tar_daily"] = 30
        
        opt_params["sl0"] = trial.suggest_float("sl0", 0.80, 0.94)
        opt_params["sl1"] = trial.suggest_float("sl1", 0.80, 0.99)
        opt_params["sl2"] = trial.suggest_float("sl2", 0.72, 1.05)
        opt_params["sl3"] = trial.suggest_float("sl3", 0.70, 1.1)
        opt_params["sl4"] = trial.suggest_float("sl4", 0.65, 1.2)
        
        opt_params["tp0"] = trial.suggest_float("tp0", 1.1, 1.6)
        opt_params["tp1"] = trial.suggest_float("tp1", 1.05, 1.8)
        opt_params["tp2"] = trial.suggest_float("tp2", 1.00, 1.8)
        opt_params["tp3"] = trial.suggest_float("tp3", 0.95, 1.9)
        opt_params["tp4"] = trial.suggest_float("tp4", 0.9, 1.95)

        params = dict(self.base_params)
        params.update(opt_params)
        return params

    def run(
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
        opt_params: dict,
    ) -> tuple[np.ndarray, ...]: 
        
        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)
        
        sl_vec = [opt_params["sl0"], opt_params["sl1"], opt_params["sl2"], opt_params["sl3"], opt_params["sl4"]]
        tp_vec = [opt_params["tp0"], opt_params["tp1"], opt_params["tp2"], opt_params["tp3"], opt_params["tp4"]]
        sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat = HelperSLTP.replicate(
            sl_vec,
            tp_vec,
            Xtr_tree,
            Xte_tree,
        )

        #logger.info(f"  Precompute -> sl {sl_tr_mat[0,:]} | tp: {tp_tr_mat[0,:]}")

        score_tr = np.ones(Xtr_tree.shape[0], dtype=np.float32)
        score_te = np.ones(Xte_tree.shape[0], dtype=np.float32)
        return mask_train, mask_test, sl_tr_mat, sl_te_mat, tp_tr_mat, tp_te_mat, score_tr, score_te

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
        
        assert timenames.index("FeatureLSTM_AdjClose") == 0, "FeatureLSTM_AdjClose is required in X*_time for MACD computation."
        assert timenames.index("FeatureLSTM_AdjHigh") == 2, "FeatureLSTM_AdjHigh is required in X*_time for ATR computation."
        assert timenames.index("FeatureLSTM_AdjLow") == 3, "FeatureLSTM_AdjLow is required in X*_time for ATR computation."
        
        mask_train, mask_test, _, _, _, _, _, _ = StratSelectedMasks().run(
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
            opt_params=self.precompute_params,
        )

        #sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
        #    ytr_tree, 
        #    ytr_tree_low, 
        #    ytr_tree_high, 
        #    ytr_tree_open,
        #    n_grid=7,
        #    spread_cost=0.0,
        #    commission=0.0,
        #)
        sl_val, tp_val = 0.88, 2.0
        sl_tr = sl_val * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = sl_val * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = tp_val * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = tp_val * np.ones(Xte_tree.shape[0], dtype=float)
        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train.mean(),
            100 * mask_test.mean(),
        )
        logger.info(f"  Precompute -> sl {sl_val} | tp: {tp_val}")

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te
    
    #------------------------------------------------------------------
    # Helper methods
    #------------------------------------------------------------------
