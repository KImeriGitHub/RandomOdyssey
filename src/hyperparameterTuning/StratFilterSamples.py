import numpy as np
import polars as pl
import pandas as pd
import datetime
import scipy
import optuna
import torch
import random

from src.hyperparameterTuning.BaseStrategy import BaseStrategy
from src.predictionModule.LoadupSamples import LoadupSamples
from src.predictionModule.FilterSamples import FilterSamples
from src.predictionModule.MachineModels import MachineModels

import logging
logger = logging.getLogger(__name__)

class StratFilterSamples(BaseStrategy):
    default_params = {
        "idxAfterPrediction": 5,
        'timesteps': 60,
        'target_option': 'last',
        "LoadupSamples_time_scaling_stretch": True,
        "LoadupSamples_time_inc_factor": 61,
        
        "Treetime_LSTM_days_to_train": 1000,
        "LSTM_units": 32,
        "LSTM_num_layers": 1,
        "LSTM_dropout": 0.003264,
        "LSTM_recurrent_dropout": 0.028886,
        "LSTM_learning_rate": 0.000135,
        "LSTM_optimizer": "adam",
        "LSTM_bidirectional": True,
        "LSTM_batch_size": 2**12,
        "LSTM_epochs": 2,
        "LSTM_l1": 0.003816,
        "LSTM_l2": 0.000290,
        "LSTM_inter_dropout": 0.001947,
        "LSTM_input_gaussian_noise": 0.001,
        "LSTM_conv1d": True,
        "LSTM_conv1d_kernel_size": 3,
        "LSTM_loss": "mse",
    
        "FilterSamples_q_up": 0.985,
        "FilterSamples_days_to_train_end": 115,
        "FilterSamples_cat_over20": True,
        "FilterSamples_cat_posOneYearReturn": False,
        "FilterSamples_cat_posFiveYearReturn": False,
        
        "FilterSamples_lincomb_epochs": 5,
        "FilterSamples_lincomb_show_progress": False,
        "FilterSamples_lincomb_featureratio": 0.5,
        "FilterSamples_lincomb_itermax": 1,
        "FilterSamples_lincomb_init_toprand":  3,
        "FilterSamples_lincomb_batch_size": 2**12,
    
        "FilterSamples_taylor_horizon_days": 50,
        "FilterSamples_taylor_roll_window_days": 10,
        "FilterSamples_taylor_weight_slope": 1.268923
    }

    def __init__(self, filter_method):
        self.filter_method = filter_method

    def sample_params(self, trial: optuna.Trial) -> dict:
        LINCOMB_SPACE = {
            "FilterSamples_days_to_train_end": ("int", 250, 350, {"step": 10}),
            "FilterSamples_lincomb_lr": ("float", 1e-4, 5e-3, {"log": False}),
            "FilterSamples_lincomb_epochs": ("int", 10, 200, {"step": 10}),
            "FilterSamples_lincomb_probs_noise_std": ("float", 0.1, 0.3, {"log": False}),
            "FilterSamples_lincomb_subsample_ratio": ("float", 0.2, 0.4, {}),
            "FilterSamples_lincomb_sharpness": ("float", 2.0, 5.0, {"log": False}),
            #"FilterSamples_lincomb_featureratio": ("float", 0.05, 0.99, {"log": True}),
            #"FilterSamples_lincomb_itermax": ("int", 1, 3, {}),
            "FilterSamples_lincomb_init_toprand": ("int", 2, 15, {})
        }
        TAYLOR_SPACE = {
            "FilterSamples_days_to_train_end": ("int", 19, 48, {"step": 1}),
            "FilterSamples_taylor_horizon_days": ("int", 24, 28, {"step": 2}),
            "FilterSamples_taylor_roll_window_days": ("int", 1, 6, {"step": 1}),
            "FilterSamples_taylor_weight_slope": ("float", 1.5, 3.5, {"log": False})
        }

        PARAMS = self.default_params.copy()
        if self.filter_method == 'lincomb':
            PARAMS.update(self.__parse_params(trial, LINCOMB_SPACE))

        elif self.filter_method == 'taylor':
            PARAMS.update(self.__parse_params(trial, TAYLOR_SPACE))

        return PARAMS

    def __parse_params(self, trial, space):
        out = {}
        for name, spec in space.items():
            kind, lo, hi, kw = spec
            sug = trial.suggest_int if kind == "int" else trial.suggest_float
            out[name] = sug(name.replace("FilterSamples_", ""), lo, hi, **kw)
        return out
    
    def score(self, 
            Xtr_tree,
            Xtr_time, 
            ytr_tree, 
            Xte_tree,
            Xte_time, 
            yte_tree,
            params: dict
        ) -> float | None:
        mask_train_pre: np.ndarray   = params.get("pre_mask_train", None)
        mask_test_pre: np.ndarray    = params.get("pre_mask_test", None)
        treenames: list[str]         = params.get("treenames", None)
        mm: MachineModels            = params.get("model_machine", None)
        meta_train: pl.DataFrame     = params.get("meta_train", None)
        meta_test: pl.DataFrame      = params.get("meta_test", None)

        if (mask_train_pre is None 
            or mask_test_pre is None
            or mm is None
            or treenames is None
            or meta_train is None
            or meta_test is None):
            raise ValueError("All score information must be provided in params.")

        fs = FilterSamples(
            Xtree_train = Xtr_tree[mask_train_pre], 
            ytree_train = ytr_tree[mask_train_pre], 
            treenames   = treenames,
            Xtree_test  = Xte_tree[mask_test_pre],
            ytree_test  = yte_tree[mask_test_pre],
            meta_train  = meta_train[mask_train_pre],
            meta_test   = meta_test[mask_test_pre],
            params      = mm.params,
        )
        if self.filter_method == 'lincomb':
            mask_train, mask_test = FilterSamples.lincomb_masks(fs)
        elif self.filter_method == 'taylor':
            mask_train, mask_test = FilterSamples.taylor_feature_masks(fs)

        score_train = fs.evaluate_mask(mask_train, meta_train['date'].filter(pl.Series(mask_train_pre)), ytr_tree[mask_train_pre])
        score_test  = fs.evaluate_mask(mask_test,  meta_test['date'].filter(pl.Series(mask_test_pre)),  yte_tree[mask_test_pre])
        logger.info(f"  Score (train) = {score_train}")
        logger.info(f"  Score (test)  = {score_test}")

        return score_test