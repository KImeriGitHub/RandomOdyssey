
def _create_filter_samples():
    import numpy as np
    import polars as pl
    import datetime
    import sys, types
    sys.modules.setdefault("torch", types.SimpleNamespace(Tensor=type("Tensor", (), {})))
    from src.predictionModule.FilterSamples import FilterSamples

    Xtree_train = np.random.rand(10, 4)
    ytree_train = np.random.rand(10)
    Xtree_test = np.random.rand(5, 4)
    dates_train = pl.Series(
        name="dates", values=[datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(10)], dtype=pl.Date
    )
    treenames = ["featA", "featCategory", "feat_lag", "Seasonal"]
    fs = FilterSamples(
        Xtree_train=Xtree_train,
        ytree_train=ytree_train,
        treenames=treenames,
        Xtree_test=Xtree_test,
        samples_dates_train=dates_train,
        ytree_test=None,
    )
    return fs


def test_separate_treefeatures():
    fs = _create_filter_samples()
    mask = fs.separate_treefeatures()
    assert mask.tolist() == [True, False, False, False]
