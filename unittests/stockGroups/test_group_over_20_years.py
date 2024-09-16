# tests/test_group_over_20_years.py
import unittest
from datetime import datetime, timedelta
import pandas as pd
from src.common.AssetData import AssetData
from src.stockGroupsService.GroupOver20Years import GroupOver20Years

class TestGroupOver20Years(unittest.TestCase):
    def test_check_asset_true(self):
        """
        Test that check_asset returns True when the asset has over 20 years of data.
        """
        # Create an AssetData object with share price data over 20 years
        start_date = datetime.today() - timedelta(days=20 * 365.25 + 10)
        dates = pd.date_range(start=start_date, periods=int(20 * 365.25) + 10)
        prices = pd.Series(100, index=dates)
        shareprice = pd.DataFrame({'Price': prices})
        asset = AssetData(ticker='TEST', shareprice=shareprice)
        group = GroupOver20Years()
        self.assertTrue(group.checkAsset(asset))

    def test_check_asset_false(self):
        """
        Test that check_asset returns False when the asset has less than 20 years of data.
        """
        # Create an AssetData object with share price data less than 20 years
        start_date = datetime.today() - timedelta(days=19 * 365.25)
        dates = pd.date_range(start=start_date, periods=int(19 * 365.25))
        prices = pd.Series(100, index=dates)
        shareprice = pd.DataFrame({'Price': prices})
        asset = AssetData(ticker='TEST', shareprice=shareprice)
        group = GroupOver20Years()
        self.assertFalse(group.checkAsset(asset))

if __name__ == '__main__':
    unittest.main()
