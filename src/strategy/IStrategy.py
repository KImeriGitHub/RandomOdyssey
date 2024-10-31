from abc import ABC, abstractmethod
from typing import Dict
from src.common.AssetData import AssetData
from src.common.Portfolio import Portfolio
import pandas as pd
import numpy as np

class IStrategy(ABC):
    def __init__(self, printBuySell):
        self.printBuySell = printBuySell

    @abstractmethod
    def apply():
        pass
    
    def sell(self,
              sellOrders: Dict,
              portfolio: Portfolio,
              currentDate: pd.Timestamp):
        if not sellOrders:
            return
        # Sell
        for ticker in sellOrders.keys():
            quantity = sellOrders[ticker]['quantity']
            price = sellOrders[ticker]['price']
            portfolio.sell(ticker, quantity, price, currentDate)

            if self.printBuySell:
                print(f"Sold {quantity} shares of {ticker} at {np.floor(price)} on date: {currentDate}.")

    def buy(self,
              buyOrders: Dict,
              portfolio: Portfolio,
              currentDate: pd.Timestamp):
        if not buyOrders:
            return
        for ticker in buyOrders.keys():
            quantity = buyOrders[ticker]['quantity']
            price = buyOrders[ticker]['price']
            portfolio.buy(ticker, quantity, price, currentDate)
            
            if self.printBuySell:
                print(f"Bought {quantity} shares of {ticker} at {np.floor(price)} on Date: {currentDate}.")