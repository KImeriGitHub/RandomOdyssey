from abc import ABC, abstractmethod
from typing import Dict, List
from src.common.Portfolio import Portfolio
from src.strategy.IStrategy import IStrategy
from src.common.AssetData import AssetData
import pandas as pd
import datetime

class ISimulation(ABC):
    @abstractmethod
    def run(self):
        pass