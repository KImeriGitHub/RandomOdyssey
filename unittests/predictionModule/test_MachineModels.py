#
#

def test_run_lgb_returns_booster():
    import numpy as np
    import lightgbm as lgb
    import sys, types
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.nn = types.SimpleNamespace(Module=type("Module", (), {}))
    torch_stub.optim = types.SimpleNamespace()
    torch_stub.utils = types.SimpleNamespace(data=types.SimpleNamespace(DataLoader=object, TensorDataset=object))
    sys.modules['torch'] = torch_stub
    sys.modules['torch.nn'] = torch_stub.nn
    sys.modules['torch.optim'] = torch_stub.optim
    sys.modules['torch.utils'] = torch_stub.utils
    sys.modules['torch.utils.data'] = torch_stub.utils.data
    from src.predictionModule.MachineModels import MachineModels

    params = {"LGB_num_boost_round": 10}
    mm = MachineModels(params)

    X_train = np.random.rand(20, 3)
    y_train = np.random.rand(20)
    X_test = np.random.rand(5, 3)
    y_test = np.random.rand(5)

    model, res = mm.run_LGB(X_train, y_train, X_test, y_test)
    assert isinstance(model, lgb.Booster)
    assert 'feature_importance' in res
    assert len(res['feature_importance']) == 3
