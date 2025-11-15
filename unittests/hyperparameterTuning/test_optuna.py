import datetime
import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import optuna
import polars as pl
import pytest

from src.hyperparameterTuning.OptunaClient import OptunaClient
from src.hyperparameterTuning.StratFilterSamples import StratFilterSamples


def _business_days(start: datetime.date, count: int) -> list[datetime.date]:
    days: list[datetime.date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += datetime.timedelta(days=1)
    return days


def _dummy_loadup_samples(n_samples: int = 40) -> SimpleNamespace:
    rng = np.random.default_rng(42)
    dates = _business_days(datetime.date(2020, 1, 1), n_samples)
    closes = np.linspace(10.0, 20.0, n_samples)
    adj_closes = closes + 0.5

    meta = pl.DataFrame(
        {
            "date": dates,
            "Close": closes,
            "AdjClose": adj_closes,
            "Open": closes + 0.2,
            "ticker": ["AAA"] * n_samples,
        }
    )

    return SimpleNamespace(
        train_Xtree=rng.normal(size=(n_samples, 3)),
        train_ytree=rng.uniform(0.8, 1.2, size=n_samples),
        train_Xtime=None,
        train_ytime=None,
        featureTreeNames=["f1", "f2", "f3"],
        featureTimeNames=[],
        meta_pl_train=meta,
    )


class _DummyStrategy:
    def __init__(self):
        self.calls = 0

    def sample_params(self, trial: optuna.Trial) -> dict:
        return {}

    def mask_precompute(self, *args, **kwargs):
        Xtr_tree = args[0]
        Xte_tree = args[3]
        return {
            "pre_mask_train": np.ones(Xtr_tree.shape[0], dtype=bool),
            "pre_mask_test": np.ones(Xte_tree.shape[0], dtype=bool),
        }

    def score(self, *args, **kwargs) -> float:
        self.calls += 1
        yte_tree = args[5]
        return float(np.mean(yte_tree))


class _DummyModel:
    def __init__(self, params: dict):
        self.params = params


class _FakeFilterSamples:
    def __init__(self, Xtree_train, ytree_train, treenames, Xtree_test, ytree_test, meta_train, meta_test, params):
        self.Xtree_train = Xtree_train
        self.ytree_train = ytree_train
        self.Xtree_test = Xtree_test
        self.ytree_test = ytree_test
        self.meta_train = meta_train
        self.meta_test = meta_test
        self.params = params

    def categorical_masks(self):
        return np.ones(self.Xtree_train.shape[0], dtype=bool), np.ones(self.Xtree_test.shape[0], dtype=bool)

    def get_recent_training_mask(self, *_):
        return np.ones(self.Xtree_train.shape[0], dtype=bool)

    def lincomb_masks(self):
        return np.ones(self.Xtree_train.shape[0], dtype=bool), np.ones(self.Xtree_test.shape[0], dtype=bool)

    def taylor_feature_masks(self):
        return self.lincomb_masks()

    def evaluate_mask(self, mask, dates, y):
        mask = np.asarray(mask, dtype=bool)
        return float(np.mean(y[mask]))


def test_optuna_client_get_slices() -> None:
    ls = _dummy_loadup_samples()
    client = OptunaClient(
        ls=ls,
        n_splits=5,
        n_test_idxdays=3,
        n_training_idxdays=10,
        rng=random.Random(0),
        model_cls=_DummyModel,
    )

    slices = client.get_slices()
    assert len(slices) == 5
    for train_slice, test_slice in slices:
        assert train_slice.stop - train_slice.start > 0
        assert test_slice.stop - test_slice.start > 0
        assert train_slice.stop <= test_slice.start


def test_optuna_client_objective_uses_strategy() -> None:
    ls = _dummy_loadup_samples()
    client = OptunaClient(
        ls=ls,
        n_splits=3,
        n_test_idxdays=2,
        n_training_idxdays=8,
        rng=random.Random(1),
        model_cls=_DummyModel,
    )
    strategy = _DummyStrategy()
    objective = client.make_objective(strategy=strategy)

    result = objective(optuna.trial.FixedTrial({}))
    assert strategy.calls == 3
    assert np.isfinite(result)


def test_strat_filter_samples_mask_precompute_shapes(monkeypatch) -> None:
    ls = _dummy_loadup_samples()
    strategy = StratFilterSamples(filter_method="lincomb")
    dummy_model = _DummyModel({"FilterSamples_days_to_train_end": 30})

    fake_module = types.ModuleType("FilterSamples")
    fake_module.FilterSamples = _FakeFilterSamples
    monkeypatch.setitem(sys.modules, "src.predictionModule.FilterSamples", fake_module)

    params = {
        "mm": dummy_model,
        "treenames": ls.featureTreeNames,
        "meta_train": ls.meta_pl_train,
    }

    masks = strategy.precompute(
        ls.train_Xtree,
        ls.train_Xtime,
        ls.train_ytree,
        ls.train_Xtree,
        ls.train_Xtime,
        ls.train_ytree,
        params=params,
    )

    assert masks["pre_mask_train"].shape[0] == ls.train_Xtree.shape[0]
    assert masks["pre_mask_test"].shape[0] == ls.train_Xtree.shape[0]


def test_strat_filter_samples_score_with_stub(monkeypatch) -> None:
    ls = _dummy_loadup_samples(20)
    strategy = StratFilterSamples(filter_method="lincomb")
    dummy_model = _DummyModel({"FilterSamples_days_to_train_end": 10})

    fake_module = types.ModuleType("FilterSamples")
    fake_module.FilterSamples = _FakeFilterSamples
    monkeypatch.setitem(sys.modules, "src.predictionModule.FilterSamples", fake_module)

    base_params = {
        "mm": dummy_model,
        "treenames": ls.featureTreeNames,
        "meta_train": ls.meta_pl_train,
    }
    mask_params = strategy.precompute(
        ls.train_Xtree,
        ls.train_Xtime,
        ls.train_ytree,
        ls.train_Xtree,
        ls.train_Xtime,
        ls.train_ytree,
        params=base_params,
    )

    score = strategy.score(
        ls.train_Xtree,
        ls.train_Xtime,
        ls.train_ytree,
        ls.train_Xtree,
        ls.train_Xtime,
        ls.train_ytree,
        params={**base_params, **mask_params},
    )

    expected = float(np.mean(ls.train_ytree))
    assert score == pytest.approx(expected)
