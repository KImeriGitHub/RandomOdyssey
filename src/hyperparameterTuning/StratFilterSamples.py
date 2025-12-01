import numpy as np
import optuna
import polars as pl
import datetime

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels
from src.hyperparameterTuning.HelperFunctions import HelperFunctions

from src.common.DataFrameTimeOperations import DataFrameTimeOperations as dfta

import logging
logger = logging.getLogger(__name__)

class StratFilterSamples(BaseStrategy):
    taylor_filter_max_days_to_consider = 1500
    expected_load_params = {
        "idxAfterPrediction": 6,
        "LoadupSamples_time_inc_factor": 1,
        "LoadupSamples_tree_scaling_standard": False,
        "LoadupSamples_time_scaling_stretch": False,
    }

    precompute_params = {
        "FilterSamples_cat_over10": True,
        "FilterSamples_cat_under5000": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        "FilterSamples_cat_highestShareholderEquity_q0.2": False,
        "FilterSamples_cat_volatility_qdown0.2": False,
        "FilterSamples_cat_volatility_qup0.8": False,
        "FilterSamples_cat_predictability_qup0.8": False,
        "FilterSamples_cat_predictability_qdown0.2": True,
    }

    base_params = {        
        "FilterSamples_lincomb_probs_noise_std": 0.001,
        "FilterSamples_lincomb_subsample_ratio": 1.0,
        "FilterSamples_lincomb_sharpness": 1.0,
        "FilterSamples_lincomb_init_toprand": 3,
        "FilterSamples_lincomb_featureratio": 0.8,
        "FilterSamples_lincomb_itermax": 1,
        
        "FilterSamples_days_to_train_end": 500,
        "FilterSamples_taylor_horizon_days": 105,
        "FilterSamples_taylor_roll_window_days": 105,
        "FilterSamples_taylor_weight_slope": 0.0,
    }

    def __init__(self, filter_method: str) -> None:
        if filter_method not in {"lincomb", "taylor"}:
            raise ValueError("filter_method must be either 'lincomb' or 'taylor'.")
        self.filter_method = filter_method
        logger.info("Initialized StratFilterSamples with filter_method: %s", self.filter_method)

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        # define params by calling `trial.suggest_*` directly
        lincomb_params = {} 
        """{
            "FilterSamples_lincomb_lr":                 trial.suggest_float("lincomb_lr", 0.01, 0.3, log=True),
            "FilterSamples_lincomb_epochs":             trial.suggest_int("lincomb_epochs", 7, 14, step=1),
            "FilterSamples_lincomb_probs_noise_std":    trial.suggest_float("lincomb_probs_noise_std", 0.012, 0.025, log=True),
            "FilterSamples_lincomb_subsample_ratio":    trial.suggest_float("lincomb_subsample_ratio", 0.1, 0.35, log=True),
            "FilterSamples_lincomb_sharpness":          trial.suggest_float("lincomb_sharpness", 0.3, 0.9),
            # "FilterSamples_lincomb_init_toprand":     trial.suggest_int("lincomb_init_toprand", 1, 4),
            # "FilterSamples_lincomb_featureratio":     trial.suggest_float("lincomb_featureratio", 0.15, 0.9),
            # "FilterSamples_lincomb_itermax":          trial.suggest_int("lincomb_itermax", 1, 3),
        }"""

        taylor_params = {
            "FilterSamples_taylor_horizon_days":        trial.suggest_int("taylor_horizon_days", 50, 400, step=25),
            "FilterSamples_taylor_roll_window_days":    trial.suggest_int("taylor_roll_window_days", 200, 600, step=25),
            "FilterSamples_taylor_weight_slope":        trial.suggest_float("taylor_weight_slope", 0.1, 2.3),
        }

        opt_params = {
            "idx_tar":                                  trial.suggest_int("idx_tar", 3, 6),
            "sl_min":                                   trial.suggest_float("sl_min", 0.89, 0.97),
            "tp_max":                                   trial.suggest_float("tp_max", 1.025, 1.07),
            "FilterSamples_days_to_train_end":          trial.suggest_int("days_to_train_end", 100, 450, step=25),
            "inc_FeatureTA":                            True, #trial.suggest_categorical("inc_FeatureTA", [True, False]),
            "inc_GroupDynamics":                        False, #trial.suggest_categorical("inc_GroupDynamics", [True, False]),
            "inc_Categorical":                          True, #trial.suggest_categorical("inc_Categorical", [True, False]),
            "inc_Financials":                           True, #trial.suggest_categorical("inc_Financials", [True, False]),
            "inc_Mathematical":                         True, #trial.suggest_categorical("inc_Mathematical", [True, False]),
            "inc_Seasonal":                             False, #trial.suggest_categorical("inc_Seasonal", [True, False]),
            "exc_lag":                                  True, #trial.suggest_categorical("exc_lag", [True, False]),
            "ytree_kind":                               "abslast", #trial.suggest_categorical("ytree_kind", ["last", "abslast"]),
            "min_n_tar":                                50,
        }

        params = dict(self.base_params)
        params.update(lincomb_params if self.filter_method == "lincomb" else taylor_params)
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
    ) -> float:        
        idx_tar = opt_params["idx_tar"]
        sl_min = opt_params["sl_min"]
        tp_max = opt_params["tp_max"]
        opt_params["FilterSamples_q_up"] =  1 - opt_params["min_n_tar"] / Xte_tree.shape[0]
        
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
            
        if not any(mask_treenames):
            return np.ones(Xte_tree.shape[0], dtype=bool), 0.88 * np.ones(Xte_tree.shape[0], dtype=float), 1.2 * np.ones(Xte_tree.shape[0], dtype=float)
            
        Xd_tr = Xtr_tree[:, mask_treenames]
        Xd_te = Xte_tree[:, mask_treenames]
        
        if opt_params["ytree_kind"] == "last":
            ytr_tree_opt = ytr_tree[:, idx_tar-1]
        elif opt_params["ytree_kind"] == "mean":
            ytr_tree_opt = np.mean(ytr_tree[:, :idx_tar], axis=1)
        elif opt_params["ytree_kind"] == "abslast":
            ytr_tree_opt = np.abs(ytr_tree[:, idx_tar-1])
        elif opt_params["ytree_kind"] == "max":
            ytr_tree_opt = np.max(ytr_tree[:, :idx_tar], axis=1)

        fs = FilterSamples(
            Xtree_train=Xd_tr,
            ytree_train=ytr_tree_opt,
            treenames=tn[mask_treenames].tolist(),
            Xtree_test=Xd_te,
            ytree_test=None,
            meta_train=meta_train,
            meta_test=meta_test,
            params=opt_params,
        )

        if self.filter_method == "lincomb":
            mask_train, mask_test = fs.lincomb_masks()
        else:
            mask_train, mask_test = fs.taylor_feature_masks()

        score_train = fs.evaluate_mask(
            mask_train,
            meta_train["date"],
            ytr_tree[:, idx_tar-1],
        )
        
        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train][:, :idx_tar], 
            ytr_tree_low[mask_train][:, :idx_tar], 
            ytr_tree_high[mask_train][:, :idx_tar], 
            ytr_tree_open[mask_train][:, :idx_tar],
            n_grid = 12,
            spread_cost=0.0,
            commission=0.0,
        )
        sl_tr = min(sl_min, sl_val) * np.ones(Xtr_tree.shape[0], dtype=float)
        sl_te = min(sl_min, sl_val) * np.ones(Xte_tree.shape[0], dtype=float)
        tp_tr = max(tp_max, tp_val) * np.ones(Xtr_tree.shape[0], dtype=float)
        tp_te = max(tp_max, tp_val) * np.ones(Xte_tree.shape[0], dtype=float)
        
        score_tr = np.random.rand(Xtr_tree.shape[0])
        score_te = np.random.rand(Xte_tree.shape[0])

        logger.info("  Score (train) = %s", score_train)

        return mask_train, mask_test, sl_tr, sl_te, tp_tr, tp_te, score_tr, score_te

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

        params = dict(self.precompute_params)

        mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree[:,-1],
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=None,
            meta_train=meta_train,
            meta_test=meta_test,
            params=params,
        )

        cat_train, cat_test = fs.categorical_masks()
        mask_train &= cat_train
        if cat_test is not None:
            mask_test &= cat_test

        if self.filter_method == "taylor":
            dates_tr = meta_train["date"].unique().sort()
            last_day = dates_tr[-1]
            start_day = last_day - datetime.timedelta(days=self.taylor_filter_max_days_to_consider)
            
            filtered_train_mask: pl.Series = (meta_train["date"] >= start_day) & (meta_train["date"] <= last_day)
            mask_train &= filtered_train_mask.fill_null(False).to_numpy()

        sl_val, tp_val, _ = HelperFunctions.optimize_sl_tp(
            ytr_tree[mask_train], 
            ytr_tree_low[mask_train], 
            ytr_tree_high[mask_train], 
            ytr_tree_open[mask_train],
            n_grid = 5,
            spread_cost=0.0000,
            commission=0.0,
        )
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