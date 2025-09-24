import logging
from typing import Dict

import numpy as np
import optuna
import polars as pl

from src.hyperparameterTuning.BaseStrategy import BaseStrategy

logger = logging.getLogger(__name__)


class StratFilterSamples(BaseStrategy):
    default_params = {
        "idxAfterPrediction": 5,
        "timesteps": 60,
        "target_option": "last",
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
        "FilterSamples_lincomb_init_toprand": 3,
        "FilterSamples_lincomb_batch_size": 2**12,
        "FilterSamples_taylor_horizon_days": 50,
        "FilterSamples_taylor_roll_window_days": 10,
        "FilterSamples_taylor_weight_slope": 1.268923,
    }

    def __init__(self, filter_method: str) -> None:
        if filter_method not in {"lincomb", "taylor"}:
            raise ValueError("filter_method must be either 'lincomb' or 'taylor'.")
        self.filter_method = filter_method

    # ------------------------------------------------------------------
    # Optuna hooks
    # ------------------------------------------------------------------
    def sample_params(self, trial: optuna.Trial) -> dict:
        lincomb_space = {
            "FilterSamples_days_to_train_end": ("int", 250, 350, {"step": 10}),
            "FilterSamples_lincomb_lr": ("float", 1e-4, 5e-3, {"log": False}),
            "FilterSamples_lincomb_epochs": ("int", 10, 200, {"step": 10}),
            "FilterSamples_lincomb_probs_noise_std": ("float", 0.1, 0.3, {"log": False}),
            "FilterSamples_lincomb_subsample_ratio": ("float", 0.2, 0.4, {}),
            "FilterSamples_lincomb_sharpness": ("float", 2.0, 5.0, {"log": False}),
            "FilterSamples_lincomb_init_toprand": ("int", 2, 15, {}),
        }
        taylor_space = {
            "FilterSamples_days_to_train_end": ("int", 19, 48, {"step": 1}),
            "FilterSamples_taylor_horizon_days": ("int", 24, 28, {"step": 2}),
            "FilterSamples_taylor_roll_window_days": ("int", 1, 6, {"step": 1}),
            "FilterSamples_taylor_weight_slope": ("float", 1.5, 3.5, {"log": False}),
        }

        params = dict(self.default_params)
        if self.filter_method == "lincomb":
            params.update(self._parse_params(trial, lincomb_space))
        else:
            params.update(self._parse_params(trial, taylor_space))
        return params

    def score(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        yte_tree,
        params: dict,
    ) -> float:
        mask_train_pre = self._ensure_mask(params.get("pre_mask_train"), Xtr_tree)
        mask_test_pre = self._ensure_mask(params.get("pre_mask_test"), Xte_tree)
        treenames = params.get("treenames")
        meta_train = params.get("meta_train")
        meta_test = params.get("meta_test")
        mm = params.get("mm") or params.get("model_machine")

        if treenames is None or meta_train is None:
            raise ValueError("Tree feature names and meta_train must be provided.")
        if mm is None or not hasattr(mm, "params"):
            raise ValueError("A MachineModels-like object with 'params' must be supplied.")

        meta_train_filtered = meta_train.filter(pl.Series(mask_train_pre))
        if meta_test is None:
            meta_test = meta_train
        meta_test_filtered = meta_test.filter(pl.Series(mask_test_pre))

        y_test = yte_tree[mask_test_pre] if yte_tree is not None else ytr_tree[mask_test_pre]

        from src.predictionModule.FilterSamples import FilterSamples

        fs = FilterSamples(
            Xtree_train=Xtr_tree[mask_train_pre],
            ytree_train=ytr_tree[mask_train_pre],
            treenames=treenames,
            Xtree_test=Xte_tree[mask_test_pre],
            ytree_test=y_test,
            meta_train=meta_train_filtered,
            meta_test=meta_test_filtered,
            params=mm.params,
        )

        if self.filter_method == "lincomb":
            mask_train, mask_test = fs.lincomb_masks()
        else:
            mask_train, mask_test = fs.taylor_feature_masks()

        score_train = fs.evaluate_mask(
            mask_train,
            meta_train_filtered["date"],
            ytr_tree[mask_train_pre],
        )
        score_test = fs.evaluate_mask(
            mask_test,
            meta_test_filtered["date"],
            y_test,
        )

        logger.info("  Score (train) = %s", score_train)
        logger.info("  Score (test)  = %s", score_test)

        return float(score_test)

    def mask_precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        yte_tree,
        params: dict,
    ) -> Dict[str, np.ndarray]:
        treenames = params.get("treenames")
        meta_train = params.get("meta_train")
        meta_test = params.get("meta_test")
        mm = params.get("mm") or params.get("model_machine")

        if treenames is None or meta_train is None or mm is None or not hasattr(mm, "params"):
            raise ValueError("treenames, meta_train and a MachineModels instance are required.")

        if meta_test is None:
            meta_test = meta_train

        cat_mask_train = np.ones(Xtr_tree.shape[0], dtype=bool)
        cat_mask_test = np.ones(Xte_tree.shape[0], dtype=bool)

        from src.predictionModule.FilterSamples import FilterSamples

        fs = FilterSamples(
            Xtree_train=Xtr_tree,
            ytree_train=ytr_tree,
            treenames=treenames,
            Xtree_test=Xte_tree,
            ytree_test=yte_tree,
            meta_train=meta_train,
            meta_test=meta_test,
            params=mm.params,
        )

        cat_train, cat_test = fs.categorical_masks()
        cat_mask_train &= cat_train
        if cat_test is not None:
            cat_mask_test &= cat_test

        recent_mask = fs.get_recent_training_mask(mm.params.get("FilterSamples_days_to_train_end"))
        mask_train_pre = cat_mask_train & recent_mask
        mask_test_pre = cat_mask_test

        logger.info(
            "  Pre-masks -> train kept: %.2f%% | test kept: %.2f%%",
            100 * mask_train_pre.mean(),
            100 * mask_test_pre.mean(),
        )

        return {
            "pre_mask_train": mask_train_pre,
            "pre_mask_test": mask_test_pre,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_params(trial: optuna.Trial, space: dict) -> dict:
        out = {}
        for name, spec in space.items():
            kind, lo, hi, kw = spec
            suggest = trial.suggest_int if kind == "int" else trial.suggest_float
            out[name] = suggest(name.replace("FilterSamples_", ""), lo, hi, **kw)
        return out

    @staticmethod
    def _ensure_mask(mask, array) -> np.ndarray:
        if mask is None:
            return np.ones(array.shape[0], dtype=bool)
        return np.asarray(mask, dtype=bool)
