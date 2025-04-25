from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut
from src.stockGroupsService.GroupFinanTo2011 import GroupFinanTo2011

import logging
logger = logging.getLogger(__name__)

class GroupSnP500FinanTo2011(IGroup):
    snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]

    def groupName(self) -> str:
        return "group_snp500_finanTo2011"

    def checkAsset(self, asset: AssetData) -> bool:
        if not GroupFinanTo2011.checkAsset(self, asset):
            return False
        
        if not asset.ticker in self.snp500tickers:
            return False
        
        return True