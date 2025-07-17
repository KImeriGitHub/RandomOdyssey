import pytest
from src.databaseService.OutsourceLoader import OutsourceLoader


def test_invalid_operator():
    with pytest.raises(NotImplementedError):
        OutsourceLoader(outsourceOperator="invalid")


def test_alpha_vantage_requires_key():
    with pytest.raises(ValueError):
        OutsourceLoader(outsourceOperator="alphaVantage")


def test_request_methods_notimplemented_yfinance():
    loader = OutsourceLoader(outsourceOperator="yfinance")
    with pytest.raises(NotImplementedError):
        loader.request_shareprice("TEST")
    with pytest.raises(NotImplementedError):
        loader.request_financials("TEST")
    with pytest.raises(NotImplementedError):
        loader.request_company_overview("TEST")
