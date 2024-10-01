from dataclasses import dataclass, field
from common.ActionCost import ActionCost
import pandas as pd
import numpy as np
from typing import Dict, List

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)  # Ticker symbol to quantity

    valueOverTime: List[pd.Timestamp, float] = field(default_factory=List)
    positionsOverTime: List[pd.Timestamp, Dict[str, float]] = field(default_factory=List)

    def updateValue(self, date: pd.Timestamp, asset_prices: Dict[str, float]):
        total_value = self.cash
        for ticker, quantity in self.positions.items():
            price = asset_prices.get(ticker, 0)
            if price == np.nan:
                continue
            total_value += quantity * price
        self.valueOverTime.append((date, total_value))

    def __updatePositions(self, date: pd.Timestamp):
        if self.positionsOverTime[-1][0] == date:
            self.positionsOverTime.pop()
        
        self.positionsOverTime.append((date, self.positions))

    def buy(self, ticker: str, quantity: float, price: float, date: pd.Timestamp):
        total_value = quantity * price
        if self.cash >= total_value + ActionCost().buy(total_value):
            self.cash -= total_value + ActionCost().buy(total_value)
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
        else:
            raise NotImplementedError("Not enough cash to buy")
        
        self.__updatePositions(date)

    def sell(self, ticker: str, quantity: float, price: float, date: pd.Timestamp):
        if self.positions.get(ticker, 0) == 0:
            return
        if self.positions[ticker] >= quantity:
            self.positions[ticker] -= quantity
            if self.positions[ticker] < 0.00001:
                del self.positions[ticker]
            total_value = quantity * price
            self.cash += total_value - ActionCost().sell(total_value)
        else:
            raise NotImplementedError("Not enough cash to buy")
        
        self.__updatePositions(date)
