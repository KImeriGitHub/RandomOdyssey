import os
import logging
from typing import Dict, List, Type

import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
import argparse
from multiprocessing import Pool, cpu_count

from src.common.AssetFileInOut import AssetFileInOut
from src.common.AssetDataService import AssetDataService
from src.common.AssetDataPolars import AssetDataPolars
from src.featureAlchemy.IFeature import IFeature
from src.featureAlchemy.MainFeature import MainFeature

from src.featureAlchemy.FeatureFourierCoeff import FeatureFourierCoeff
from src.featureAlchemy.FeatureCategory import FeatureCategory
from src.featureAlchemy.FeatureFinancialData import FeatureFinancialData
from src.featureAlchemy.FeatureMathematical import FeatureMathematical
from src.featureAlchemy.FeatureSeasonal import FeatureSeasonal
from src.featureAlchemy.FeatureTA import FeatureTA
from src.featureAlchemy.FeatureGroupDynamic import FeatureGroupDynamic

FCa = FeatureCategory
FFD = FeatureFinancialData
FM  = FeatureMathematical
FS  = FeatureSeasonal
FT  = FeatureTA
FGD = FeatureGroupDynamic
FFC = FeatureFourierCoeff

feature_classes = [FCa, FFD, FM, FS, FT, FGD, FFC]
feature_classes_noFourier = [FCa, FFD, FM, FS, FT, FGD]

formatted_date = dt.now().strftime("%d%b%y_%H%M").lower()
logging.basicConfig(
    filename=f'logs/output_generating_features_{formatted_date}.txt',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

## VARIABLES
groups_features = {
    #"group_debug": (feature_classes, 'Tree'),
    #"group_snp500_finanTo2011": (feature_classes, 'Tree'),
    #"group_finanTo2011": (feature_classes_noFourier, 'Tree'),
    "group_over20Years": ([FCa, FM, FS, FT, FGD], 'Tree'),
}
lag_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]
month_horizons = [1, 2, 4, 6, 8, 12]
params = {
    "idxLengthOneMonth": 21,
    "fouriercutoff": 5,
    "multFactor": 8,
    "timesteps": 90,
    'lagList': lag_list,
    'monthHorizonList': month_horizons,
}
n_workers = 6 # For parallel processing, set to number of CPU cores or desired number of workers

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
    group_type: str,
    feature_classes: List[Type[IFeature]],
    label: str,
    start: datetime.date,
    end: datetime.date,
    params: dict,
):
    fm = MainFeature(
        assets,
        feature_classes,
        start,
        end,
        params=params,
    )
    meta_feat, arr_feat, names_feat = fm.get_features()

    # Ensure output directory exists
    out_dir = "src/featureAlchemy/bin"
    os.makedirs(out_dir, exist_ok=True)

    # Save features
    prefix = f"{group_type}Features"
    features_path = os.path.join(out_dir, f"{prefix}_{label}_{group}.npz")
    np.savez_compressed(
        features_path,
        meta_feat=meta_feat,
        featuresArr=arr_feat,
        featuresNames=names_feat,
    )
    logging.info("Saved features to %s", features_path)

def process_year(year):
    label = str(year)
    for group, (feature_classes, group_type) in groups_features.items():
        assets_pl = load_assets(group)
        start_date = pd.Timestamp(year, 1, 1).date()

        if year < dt.now().year:
            end_date = pd.Timestamp(year, 12, 31).date()
        else:
            last_dates = {
                ticker: df.shareprice["Date"].item(-1)
                for ticker, df in assets_pl.items()
            }
            end_date = max(last_dates.values())

        logger.info("Processing %s for group %s", label, group)
        t0 = dt.now()
        try:
            process_period(
                assets_pl,
                group,
                group_type,
                feature_classes,
                label,
                start_date,
                end_date,
                params,
            )
        except Exception as e:
            logger.error("Error processing %s for group %s: %s", label, group, e)
            continue
        logger.info("Processed %s in %.1f s",
                    label, (dt.now() - t0).total_seconds())

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
    
    # spawn one worker per CPU (or cap it)
    with Pool(processes=n_workers) as pool:
        pool.map(process_year, years)