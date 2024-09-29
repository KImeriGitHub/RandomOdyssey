from dataclasses import dataclass, field
from common.ActionCost import ActionCost
import pandas as pd
import numpy as np
from typing import Dict

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)  # Ticker symbol to quantity
    history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=['Date', 'Value']))

    def update_value(self, date: pd.Timestamp, asset_prices: Dict[str, float]):
        total_value = self.cash
        for ticker, quantity in self.positions.items():
            price = asset_prices.get(ticker.lower(), asset_prices.get(ticker.upper(),0))
            if price == np.nan:
                continue
            total_value += quantity * price
        self.history = pd.concat([self.history, pd.DataFrame({'Date': [date], 'Value': [total_value]})], ignore_index=True)

    def buy(self, ticker: str, quantity: float, price: float):
        total_value = quantity * price
        if self.cash >= total_value + ActionCost().buy(total_value):
            self.cash -= total_value + ActionCost().buy(total_value)
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
        else:
            raise ValueError("Not enough cash to buy")

    def sell(self, ticker: str, quantity: float, price: float):
        if self.positions.get(ticker, 0) == 0:
            return
        if self.positions[ticker] >= quantity:
            self.positions[ticker] -= quantity
            if self.positions[ticker] < 0.00001:
                del self.positions[ticker]
            total_value = quantity * price
            self.cash += total_value - ActionCost().sell(total_value)
        else:
            raise ValueError("Not enough shares to sell")
