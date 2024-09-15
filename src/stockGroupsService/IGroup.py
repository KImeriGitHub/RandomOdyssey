from abc import ABC, abstractmethod
from src.common.AssetData import AssetData

class IGroup(ABC):
    @abstractmethod
    def groupName(self) -> str:
        pass

    @abstractmethod
    def checkAsset(self, asset: AssetData) -> bool:
        pass
