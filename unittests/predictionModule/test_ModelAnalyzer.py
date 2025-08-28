#
#

def test_print_label_distribution(caplog):
    import numpy as np
    import logging
    from src.predictionModule.ModelAnalyzer import ModelAnalyzer

    arr1 = np.array([0, 1, 1])
    arr2 = np.array([1, 1, 0, 0])
    with caplog.at_level(logging.INFO):
        ModelAnalyzer.print_label_distribution(arr1, arr2)
    assert "Label" in caplog.text


def test_print_classification_metrics(caplog):
    import numpy as np
    import logging
    from src.predictionModule.ModelAnalyzer import ModelAnalyzer

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([[0.7,0.3],[0.2,0.8],[0.6,0.4],[0.8,0.2]])
    with caplog.at_level(logging.INFO):
        ModelAnalyzer.print_classification_metrics(y_true, y_pred, y_prob)
    assert "Overall Accuracy" in caplog.text


def test_print_feature_importance_LGBM(caplog):
    import numpy as np
    import lightgbm as lgb
    import logging
    from src.predictionModule.ModelAnalyzer import ModelAnalyzer

    X = np.random.rand(20,3)
    y = np.random.rand(20)
    dataset = lgb.Dataset(X, label=y)
    model = lgb.train({}, dataset, num_boost_round=1)
    with caplog.at_level(logging.INFO):
        ModelAnalyzer.print_feature_importance_LGBM(model, ["a","b","c"], n_feature=2)
    assert "Top 2 Feature Importances" in caplog.text


def test_print_model_results(caplog):
    import logging
    from src.predictionModule.ModelAnalyzer import ModelAnalyzer

    preds = [0.1, 0.2, 0.3]
    rets = [0.05, 0.1, 0.15]
    with caplog.at_level(logging.INFO):
        ModelAnalyzer.print_model_results(preds, rets)
    assert "Correlation coefficient" in caplog.text
