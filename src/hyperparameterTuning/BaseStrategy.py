import optuna
import numpy as np
import polars as pl
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Any

class BaseStrategy(ABC):
    """Interface for Optuna optimisation strategies."""

    # To check the loaded data
    expected_load_params: ClassVar[Dict[str, Any]]
    
    # params needed for precomputation
    precompute_params: ClassVar[Dict[str, Any]]

    # params needed for scoring
    base_params: ClassVar[Dict[str, Any]]

    @abstractmethod
    def sample_params(self, trial: optuna.Trial) -> dict:
        """Sample hyperparameters using the provided Optuna trial object."""

    @abstractmethod
    def run(
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
        opt_params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the model with the given parameters and return a score."""

    @abstractmethod
    def precompute(
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pre-compute information required before scoring and return masks."""
