#
#

def test_determine_idx_after_precedence():
    import datetime
    import polars as pl
    from src.predictionModule.LoadupSamples import LoadupSamples
    ls = LoadupSamples(
        train_start_date=datetime.date(2023, 1, 1),
        test_dates=[datetime.date(2023, 1, 10)],
        group="TEST",
        params={"daysAfterPrediction": 5, "idxAfterPrediction": 3},
    )
    meta = pl.DataFrame({"date": [datetime.date(2023, 1, 1)]})
    res = ls._LoadupSamples__determine_idx_after(meta)
    assert res == 3


def test_calc_trading_days():
    import datetime
    import polars as pl
    from src.predictionModule.LoadupSamples import LoadupSamples
    ls = LoadupSamples(
        train_start_date=datetime.date(2023, 1, 1),
        test_dates=[datetime.date(2023, 1, 10)],
        group="TEST",
    )
    meta = pl.DataFrame({"date": [datetime.date(2023, 1, 1)]})
    days = ls._LoadupSamples__calc_trading_days(meta, 7)
    assert days == 4


def test_scale_time_stretch_bounds():
    import numpy as np
    import datetime
    from src.predictionModule.LoadupSamples import LoadupSamples
    ls = LoadupSamples(
        train_start_date=datetime.date(2023, 1, 1),
        test_dates=[datetime.date(2023, 1, 10)],
        group="TEST",
    )
    X = np.array([
        [[0.2, 0.4], [0.5, 0.5], [0.8, 0.6]],
        [[0.1, 0.3], [0.5, 0.5], [0.9, 0.7]],
    ], dtype=float)
    res = ls._LoadupSamples__scale_time_stretch(X)
    assert res.shape == X.shape
    assert np.all(res >= 0) and np.all(res <= 1)
    assert np.allclose(res[:, 1, :], 0.5)


def test_remove_nan_samples_train_tree():
    import numpy as np
    import datetime
    import polars as pl
    from src.predictionModule.LoadupSamples import LoadupSamples
    ls = LoadupSamples(
        train_start_date=datetime.date(2023, 1, 1),
        test_dates=[datetime.date(2023, 1, 10)],
        group="TEST",
    )
    ls.group_type = "Tree"
    ls.train_Xtree = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    ls.train_ytree = np.array([1.0, 2.0, np.nan])
    ls.meta_pl_train = pl.DataFrame({"date": [datetime.date(2023,1,1), datetime.date(2023,1,2), datetime.date(2023,1,3)]})
    ls._LoadupSamples__remove_nan_samples_train()
    assert ls.train_Xtree.shape[0] == 1
    assert ls.train_ytree.shape[0] == 1
    assert ls.meta_pl_train.height == 1
