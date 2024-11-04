from src.simulation.CollectionSimulations import CollectionSimulations
from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
import numpy as np
from typing import Dict
import gc
import time

assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_over20years")
# Convert to Polars for speedup
assetspl: Dict[str, AssetDataPolars] = {}
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)

## FourierML

## Quadratic Ascend
"""
slrList = np.arange(0.77, 0.95, 0.005).round(3).tolist()
nmon = np.arange(1, 7, 1).round(0).tolist()
nmonvar = np.arange(1, 13, 1).round(0).tolist()
nchoice = np.arange(2, 3, 1).round(0).tolist()

for nc in nchoice:
    for nm in nmon:
        for nmv in nmonvar:
            if (nc ==1 and nm<3) or (nc ==1 and nmv<10 and nm==3):
                continue
            print(f"Choices: {nc},   Months: {nm},   Months Var: {nmv}")
            for slr in slrList:
                CollectionSimulations.QuadraticAscend(assets = assetspl, 
                                                      stoplossratio = slr,
                                                      num_choices=nc,
                                                      num_months=nm,
                                                      num_months_var=nmv)
                gc.collect()
                time.sleep(0.1)
                """
