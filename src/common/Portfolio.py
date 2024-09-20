from dataclasses import dataclass, field
import pandas as pd
from typing import Dict
import datetime

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)  # Ticker symbol to quantity
    history: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=['Date', 'Value']))

    def update_value(self, date: pd.Timestamp, asset_prices: Dict[str, float]):
        total_value = self.cash
        for ticker, quantity in self.positions.items():
            price = asset_prices.get(ticker.lower(), asset_prices.get(ticker.upper(),0))
            total_value += quantity * price
        self.history = pd.concat([self.history, pd.DataFrame({'Date': [date], 'Value': [total_value]})], ignore_index=True)

    def buy(self, ticker: str, quantity: float, price: float):
        total_cost = quantity * price
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
        else:
            raise ValueError("Not enough cash to buy")

    def sell(self, ticker: str, quantity: float, price: float):
        if self.positions.get(ticker, 0) >= quantity:
            self.positions[ticker] -= quantity
            if self.positions[ticker] == 0:
                del self.positions[ticker]
            self.cash += quantity * price
        else:
            raise ValueError("Not enough shares to sell")
