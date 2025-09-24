import optuna
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Interface for Optuna optimisation strategies."""

    @abstractmethod
    def sample_params(self, trial: optuna.Trial) -> dict:
        """Sample hyperparameters using the provided Optuna trial object."""

    @abstractmethod
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
        """Evaluate the model with the given parameters and return a score."""

    @abstractmethod
    def mask_precompute(
        self,
        Xtr_tree,
        Xtr_time,
        ytr_tree,
        Xte_tree,
        Xte_time,
        yte_tree,
        params: dict,
    ) -> dict:
        """Pre-compute information required before scoring and return it as a dict."""
