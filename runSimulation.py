from src.simulation.CollectionSimulations import CollectionSimulations
from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetDataPolars import AssetDataPolars
from src.common.AssetDataService import AssetDataService
import numpy as np
from typing import Dict
import gc
import time

assets=AssetFileInOut("src/stockGroups/bin").loadDictFromFile("group_snp500_finanTo2011")
# Convert to Polars for speedup
assetspl: Dict[str, AssetDataPolars] = {}
for ticker, asset in assets.items():
    assetspl[ticker]= AssetDataService.to_polars(asset)

res = np.zeros(50)
for i in range(len(res)):
    res[i] = CollectionSimulations.FreefallAndHighVol(
        assets = assetspl, 
        num_choices=5,
    )

print(np.mean(res))
"""
slrList = np.arange(0.875, 0.88, 0.005).round(3).tolist()
nmon = np.arange(6, 7, 1).round(0).tolist()
nmonvar = np.arange(12, 13, 1).round(0).tolist()
nchoice = np.arange(5, 6, 1).round(0).tolist()
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