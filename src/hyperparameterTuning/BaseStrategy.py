import optuna
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample_params(self, trial: optuna.Trial) -> dict:
        """Sample hyperparameters using the provided Optuna trial object."""
        pass

    @abstractmethod
    def score(self, 
        Xtr_tree,
        Xtr_time, 
        ytr_tree, 
        Xte_tree,
        Xte_time, 
        yte_tree,
        params: dict
    ) -> float:
        """Evaluate the model with the given parameters and return a score."""
        pass

    @abstractmethod
    def mask_precompute(self, 
        Xtr_tree,
        Xtr_time, 
        ytr_tree, 
        Xte_tree,
        Xte_time, 
        yte_tree,
        params: dict
    ) -> float:
        """Evaluate the model with the given parameters and return a score."""
        pass