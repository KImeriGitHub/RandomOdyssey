from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut
from src.stockGroupsService.GroupOver20Years import GroupOver20Years

class GroupSnP500NAS100Over20Years(IGroup):
    snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]
    nas100tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("nas100.yaml")["nas100tickers"]

    def groupName(self) -> str:
        return "group_nas100snp500_over20years"

    def checkAsset(self, asset: AssetData) -> bool:
        if not GroupOver20Years.checkAsset(self, asset):
            return False
        
        if not asset.ticker in self.snp500tickers:
            return False
        
        if not asset.ticker in self.nas100tickers:
            return False
        
        return True