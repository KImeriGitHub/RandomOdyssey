import optuna
import numpy as np
import polars as pl
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Interface for Optuna optimisation strategies."""

    base_params: dict = {}

    @abstractmethod
    def sample_params(self, trial: optuna.Trial) -> dict:
        """Sample hyperparameters using the provided Optuna trial object."""

    @abstractmethod
    def score(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        yte_tree: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame,
        meta_test: pl.DataFrame,
        opt_params: dict,
    ) -> float:
        """Evaluate the model with the given parameters and return a score."""

    @abstractmethod
    def mask_precompute(
        self,
        Xtr_tree: np.ndarray,
        Xtr_time: np.ndarray,
        ytr_tree: np.ndarray,
        Xte_tree: np.ndarray,
        Xte_time: np.ndarray,
        yte_tree: np.ndarray,
        treenames: list[str],
        timenames: list[str],
        meta_train: pl.DataFrame,
        meta_test: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pre-compute information required before scoring and return masks."""
