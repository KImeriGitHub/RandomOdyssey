from src.common.AssetData import AssetData
from src.stockGroupsService.IGroup import IGroup
from src.common.YamlTickerInOut import YamlTickerInOut
from src.stockGroupsService.Checks import Checks

class GroupSnP500NAS100Over20Years(IGroup):
    snp500tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("snp500.yaml")["snp500tickers"]
    nas100tickers = YamlTickerInOut("src/tickerSelection").loadFromFile("nas100.yaml")["nas100tickers"]

    def groupName(self) -> str:
        return "group_nas100snp500_over20years"

    def checkAsset(self, asset: AssetData) -> bool:
        if not Checks.checkOverYear(asset, 2004):
            return False
        
        if not (asset.ticker in self.snp500tickers or asset.ticker in self.nas100tickers):
            return False
        
        return True