from src.simulation.ISimulation import ISimulation
from src.common.AssetData import AssetData
import pandas as pd
import datetime

class SimulateAssetAnalysis(ISimulation):
    def __init__(self, asset: AssetData, analysis_function, start_date: datetime.datetime, end_date: datetime.datetime):
        self.asset = asset
        self.analysis_function = analysis_function
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        for date in dates:
            price_data = self.asset.shareprice.loc[self.asset.shareprice.index == date]
            if not price_data.empty:
                self.analysis_function(price_data)