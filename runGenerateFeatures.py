import os
import logging
from typing import Dict, List

import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import argparse

from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetDataService import AssetDataService
from src.common.AssetDataPolars import AssetDataPolars
from src.featureAlchemy.FeatureMain import FeatureMain

from src.featureAlchemy.FeatureFourierCoeff import FeatureFourierCoeff
from src.featureAlchemy.FeatureCategory import FeatureCategory
from src.featureAlchemy.FeatureFinancialData import FeatureFinancialData
from src.featureAlchemy.FeatureMathematical import FeatureMathematical
from src.featureAlchemy.FeatureSeasonal import FeatureSeasonal
from src.featureAlchemy.FeatureTA import FeatureTA
from src.featureAlchemy.FeatureGroupDynamic import FeatureGroupDynamic

FCo = FeatureFourierCoeff.__name__
FCa = FeatureCategory.__name__
FD = FeatureFinancialData.__name__
FM = FeatureMathematical.__name__
FS = FeatureSeasonal.__name__
FT = FeatureTA.__name__
FGD = FeatureGroupDynamic.__name__

feature_classes = [FCo, FCa, FD, FM, FS, FT, FGD]

formatted_date = dt.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f'logs/output_generating_features_{formatted_date}.txt',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

## VARIABLES
groups_features = {
    "group_debug": feature_classes,
    "group_snp500_finanTo2011": feature_classes,
}
lag_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]
month_horizons = [1, 2, 4, 6, 8, 12]
params = {
    "idxLengthOneMonth": 21,
    "fouriercutoff": 5,
    "multFactor": 8,
    "timesteps": 90,
}

## CODE FOR GENERATING FEATURES
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
    feature_classes: List[str],
    label: str,
    start: datetime.date,
    end: datetime.date,
    lag_list: List[int],
    month_horizons: List[int],
    params: dict,
):
    fm = FeatureMain(
        assets,
        feature_classes,
        start,
        end,
        lagList=lag_list,
        monthHorizonList=month_horizons,
        params=params,
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
    parser = argparse.ArgumentParser(description="Year to form features")
    parser.add_argument(
        "--year", type=int, default=None, help="Single year to form features"
    )
    parser.add_argument(
        "--min-year", type=int, default=None, help="Start year; processes through current year"
    )
    args = parser.parse_args()

    years = []
    if args.min_year is not None:
        years = list(range(args.min_year, dt.now().year + 1))
    elif args.year is not None:
        years = [args.year]
    else:
        years = [dt.now().year]

    for year in years:
        label = str(year)
        for group, feature_classes in groups_features.items():
            label = str(year)
            assets_pl = load_assets(group)
            start_date = pd.Timestamp(year, 1, 1).date()
            
            end_date = 0
            if year < dt.now().year:
                end_date = pd.Timestamp(year, 12, 31).date()
            else:
                # Use Polars max to safely get last date
                first_ticker = list(assets_pl.keys())[0]
                date = assets_pl[first_ticker].shareprice["Date"].item(-1)
                last_ts = pd.Timestamp(date)
                end_date = last_ts.date()
            
            logger.info("Processing %s for group %s", label, group)
            starttime = dt.now()
            process_period(
                assets_pl,
                group,
                feature_classes,
                label,
                start_date,
                end_date,
                lag_list,
                month_horizons,
                params,
            )
            endtime = dt.now()
            logger.info("Processed %s in %s seconds", label, (endtime - starttime).total_seconds())