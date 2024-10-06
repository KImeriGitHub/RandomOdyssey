from dataclasses import dataclass, field
from common.ActionCost import ActionCost
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)  # Ticker symbol to quantity

    valueOverTime: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    positionsOverTime: List[Tuple[pd.Timestamp, Dict[str, float]]] = field(default_factory=list)

    def updateValue(self, date: pd.Timestamp, asset_prices: Dict[str, float]):
        total_value = self.cash
        for ticker, quantity in self.positions.items():
            price = asset_prices.get(ticker, 0)
            total_value += quantity * price
        self.valueOverTime.append((date, total_value))

    def __updatePositions(self, date: pd.Timestamp):
        # Ensure we copy the positions to avoid referencing the same dict
        if self.positionsOverTime and self.positionsOverTime[-1][0] == date:
            # Replace the last entry if the date is the same
            self.positionsOverTime[-1] = (date, self.positions.copy())
        else:
            # Append a new entry with the current positions
            self.positionsOverTime.append((date, self.positions.copy()))


    def buy(self, ticker: str, quantity: float, price: float, date: pd.Timestamp):
        total_value = quantity * price
        if self.cash >= total_value + ActionCost().buy(total_value):
            self.cash -= total_value + ActionCost().buy(total_value)
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
        else:
            raise NotImplementedError("Not enough cash to buy")
        
        self.__updatePositions(date)

    def sell(self, ticker: str, quantity: float, price: float, date: pd.Timestamp):
        if self.positions[ticker] >= quantity:
            self.positions[ticker] -= quantity
            if self.positions[ticker] < 0.00001:
                del self.positions[ticker]
            total_value = quantity * price
            self.cash += total_value - ActionCost().sell(total_value)
        else:
            raise NotImplementedError("Not enough quantity to sell")
        
        self.__updatePositions(date)
