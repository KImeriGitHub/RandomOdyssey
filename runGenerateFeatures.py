import os
import logging
from typing import Dict, List

import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt

from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetDataService import AssetDataService
from src.common.AssetDataPolars import AssetDataPolars
from src.featureAlchemy.FeatureMain import FeatureMain

# Configure logging for better traceability
formatted_date = dt.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f'output_generating_features_{formatted_date}.txt',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_assets(group_name: str, base_path: str = "src/stockGroups/bin") -> Dict[str, AssetDataPolars]:
    """
    Load assets for a given group and convert to Polars format.
    """
    raw = AssetFileInOut(base_path).loadDictFromFile(group_name)
    assets_polars: Dict[str, AssetDataPolars] = {}
    for ticker, asset in raw.items():
        assets_polars[ticker] = AssetDataService.to_polars(asset)
    return assets_polars


def process_period(
    assets: Dict[str, AssetDataPolars],
    group: str,
    label: str,
    start: datetime.date,
    end: datetime.date,
    lag_list: List[int],
    month_horizons: List[int],
    params: dict,
    logger: logging.Logger,
):
    """
    Compute and save both tree and time features for a given time period.
    """
    
    fm = FeatureMain(
        assets,
        start,
        end,
        lagList=lag_list,
        monthHorizonList=month_horizons,
        params=params,
        logger=logger,
    )
    meta_tree, arr_tree, names_tree = fm.getTreeFeatures()
    meta_time, arr_time, names_time = fm.getTimeFeatures()

    # Ensure output directory exists
    out_dir = "src/featureAlchemy/bin"
    os.makedirs(out_dir, exist_ok=True)

    # Save tree features
    tree_path = os.path.join(out_dir, f"TreeFeatures_{label}_{group}.npz")
    np.savez_compressed(
        tree_path,
        meta_tree=meta_tree,
        treeFeatures=arr_tree,
        treeFeaturesNames=names_tree,
    )
    logging.info("Saved tree features to %s", tree_path)

    # Save time features
    time_path = os.path.join(out_dir, f"TimeFeatures_{label}_{group}.npz")
    np.savez_compressed(
        time_path,
        meta_time=meta_time,
        timeFeatures=arr_time,
        timeFeaturesNames=names_time,
    )
    logging.info("Saved time features to %s", time_path)
    
if __name__ == "__main__":
    groups = [
        "group_debug",
        "group_snp500_finanTo2011",
        "group_finanTo2011",
    ]
    years = np.arange(2014, 2025)
    lag_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]
    month_horizons = [1, 2, 4, 6, 8, 12]
    params = {
        "idxLengthOneMonth": 21,
        "fouriercutoff": 5,
        "multFactor": 8,
        "timesteps": 90,
    }

    for group in groups:
        assets_pl = load_assets(group)
        # Yearly periods
        for year in years:
            label = str(year)
            start_date = pd.Timestamp(year, 1, 1).date()
            end_date = pd.Timestamp(year, 12, 31).date()
            
            logger.info("Processing %s for group %s", label, group)
            starttime = dt.now()
            process_period(
                assets_pl,
                group,
                label,
                start_date,
                end_date,
                lag_list,
                month_horizons,
                params,
                logger=logger,
            )
            endtime = dt.now()
            logger.info("Processed %s in %s seconds", label, (endtime - starttime).total_seconds())

        first_ticker = list(assets_pl.keys())[0]
        date = assets_pl[first_ticker].shareprice["Date"].item(-1)
        # Use Polars max to safely get last date
        last_ts = pd.Timestamp(date)
        label = f"2025_to_{last_ts.date()}"
        process_period(
            assets_pl,
            group,
            label,
            pd.Timestamp(2025, 1, 1).date(),
            last_ts,
            lag_list,
            month_horizons,
            params,
        )